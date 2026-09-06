// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/source_context.h"

#include <string.h>

#define IREE_HAL_AMDGPU_LOOM_SITE_TABLE_MAGIC 0x5449534Cu
#define IREE_HAL_AMDGPU_LOOM_SITE_TABLE_VERSION 1u
#define IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_LENGTH 32u
#define IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_LENGTH 48u

enum iree_hal_amdgpu_loom_site_table_header_offset_e {
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_MAGIC_OFFSET = 0,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_VERSION_OFFSET = 4,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_HEADER_LENGTH_OFFSET = 5,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_RECORD_LENGTH_OFFSET = 6,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_ROW_COUNT_OFFSET = 8,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_STRING_TABLE_OFFSET_OFFSET = 12,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_STRING_TABLE_LENGTH_OFFSET = 16,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_PAYLOAD_DATA_OFFSET_OFFSET = 20,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_PAYLOAD_DATA_LENGTH_OFFSET = 24,
};

enum iree_hal_amdgpu_loom_site_table_record_offset_e {
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SITE_ID_OFFSET = 0,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_OP_KIND_OFFSET = 4,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_FLAGS_OFFSET = 8,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_PAYLOAD_OFFSET_OFFSET = 12,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_PAYLOAD_LENGTH_OFFSET = 16,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET = 20,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET = 24,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_START_LINE_OFFSET = 28,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_START_COLUMN_OFFSET = 32,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_END_LINE_OFFSET = 36,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_END_COLUMN_OFFSET = 40,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET = 44,
};

enum iree_hal_amdgpu_loom_site_table_record_flag_bits_e {
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_PAYLOAD = 1u << 0,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION = 1u << 1,
};

enum iree_hal_amdgpu_loom_site_table_source_kind_e {
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_SOURCE_KIND_NONE = 0,
  IREE_HAL_AMDGPU_LOOM_SITE_TABLE_SOURCE_KIND_FILE = 1,
};

enum iree_hal_amdgpu_loom_op_kind_e {
  IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_ACCESS = (0x1Du << 8) | 0u,
  IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_VALUE = (0x1Du << 8) | 1u,
  IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_OP = (0x1Du << 8) | 2u,
  IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_LAYOUT = (0x1Du << 8) | 3u,
  IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_RACE_ACCESS = (0x1Du << 8) | 4u,
  IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_RACE_SYNC = (0x1Du << 8) | 5u,
};

static iree_string_view_t iree_hal_amdgpu_source_context_loom_op_kind_name(
    uint32_t op_kind) {
  switch (op_kind) {
    case IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_ACCESS:
      return IREE_SV("sanitizer.assert.access");
    case IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_VALUE:
      return IREE_SV("sanitizer.assert.value");
    case IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_OP:
      return IREE_SV("sanitizer.assert.op");
    case IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_ASSERT_LAYOUT:
      return IREE_SV("sanitizer.assert.layout");
    case IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_RACE_ACCESS:
      return IREE_SV("sanitizer.race.access");
    case IREE_HAL_AMDGPU_LOOM_OP_SANITIZER_RACE_SYNC:
      return IREE_SV("sanitizer.race.sync");
    default:
      return iree_string_view_empty();
  }
}

static uint8_t iree_hal_amdgpu_source_context_load_u8(const uint8_t* data,
                                                      iree_host_size_t offset) {
  return data[offset];
}

static uint16_t iree_hal_amdgpu_source_context_load_u16(
    const uint8_t* data, iree_host_size_t offset) {
  return iree_unaligned_load_le_u16(data + offset);
}

static uint32_t iree_hal_amdgpu_source_context_load_u32(
    const uint8_t* data, iree_host_size_t offset) {
  return iree_unaligned_load_le_u32(data + offset);
}

static bool iree_hal_amdgpu_source_context_span_in_bounds(
    iree_host_size_t offset, iree_host_size_t length,
    iree_host_size_t container_length) {
  return offset <= container_length && length <= container_length - offset;
}

static iree_status_t
iree_hal_amdgpu_source_context_validate_sanitizer_site_table_record(
    const iree_hal_amdgpu_source_context_site_table_t* table,
    uint32_t record_ordinal) {
  const uint8_t* record =
      table->data + IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_LENGTH +
      ((iree_host_size_t)record_ordinal * table->record_length);
  const uint32_t site_id = iree_hal_amdgpu_source_context_load_u32(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SITE_ID_OFFSET);
  if (IREE_UNLIKELY(site_id != record_ordinal)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table has non-dense site id %u "
        "at row %u",
        site_id, record_ordinal);
  }

  const uint32_t known_flags =
      IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_PAYLOAD |
      IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION;
  const uint32_t flags = iree_hal_amdgpu_source_context_load_u32(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_FLAGS_OFFSET);
  if (IREE_UNLIKELY((flags & ~known_flags) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table row %u has unsupported "
        "flags 0x%08x",
        record_ordinal, flags);
  }

  const uint32_t payload_offset = iree_hal_amdgpu_source_context_load_u32(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_PAYLOAD_OFFSET_OFFSET);
  const uint32_t payload_length = iree_hal_amdgpu_source_context_load_u32(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_PAYLOAD_LENGTH_OFFSET);
  if (iree_any_bit_set(flags,
                       IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_PAYLOAD)) {
    if (IREE_UNLIKELY(!iree_hal_amdgpu_source_context_span_in_bounds(
            payload_offset, payload_length, table->payload_data_length))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU source context sanitizer site table row %u has out-of-range "
          "payload span",
          record_ordinal);
    }
  } else if (IREE_UNLIKELY(payload_length != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table row %u has payload bytes "
        "without a payload flag",
        record_ordinal);
  }

  const uint32_t source_name_offset = iree_hal_amdgpu_source_context_load_u32(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET);
  const uint32_t source_name_length = iree_hal_amdgpu_source_context_load_u32(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET);
  const uint16_t source_kind = iree_hal_amdgpu_source_context_load_u16(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET);
  if (iree_any_bit_set(
          flags, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION)) {
    if (IREE_UNLIKELY(source_kind !=
                      IREE_HAL_AMDGPU_LOOM_SITE_TABLE_SOURCE_KIND_FILE)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU source context sanitizer site table row %u has unsupported "
          "source kind %u",
          record_ordinal, source_kind);
    }
    if (IREE_UNLIKELY(!iree_hal_amdgpu_source_context_span_in_bounds(
            source_name_offset, (iree_host_size_t)source_name_length + 1,
            table->string_table_length))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU source context sanitizer site table row %u has out-of-range "
          "source name",
          record_ordinal);
    }
    const uint8_t* source_name =
        table->data + table->string_table_offset + source_name_offset;
    if (IREE_UNLIKELY(source_name[source_name_length] != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU source context sanitizer site table row %u source name is "
          "not NUL-terminated",
          record_ordinal);
    }
  } else if (IREE_UNLIKELY(source_kind !=
                           IREE_HAL_AMDGPU_LOOM_SITE_TABLE_SOURCE_KIND_NONE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table row %u has source kind %u "
        "without a source-location flag",
        record_ordinal, source_kind);
  }

  return iree_ok_status();
}

void iree_hal_amdgpu_source_context_initialize(
    uint64_t executable_id, const uint64_t code_object_hash[2],
    iree_host_size_t physical_device_count,
    iree_hal_amdgpu_loaded_code_object_range_t* loaded_code_object_ranges,
    iree_hal_amdgpu_source_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  memset(out_context, 0, sizeof(*out_context));
  out_context->executable_id = executable_id;
  if (code_object_hash) {
    out_context->code_object_hash[0] = code_object_hash[0];
    out_context->code_object_hash[1] = code_object_hash[1];
  }
  out_context->physical_device_count = physical_device_count;
  out_context->loaded_code_object_ranges = loaded_code_object_ranges;
  iree_atomic_store(&out_context->sanitizer_site_table_published, 0,
                    iree_memory_order_relaxed);
}

iree_status_t iree_hal_amdgpu_source_context_set_loaded_code_object_range(
    iree_hal_amdgpu_source_context_t* context,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_loaded_code_object_range_t range) {
  IREE_ASSERT_ARGUMENT(context);
  if (IREE_UNLIKELY(physical_device_ordinal >=
                    context->physical_device_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU source context physical device ordinal %" PRIhsz
        " exceeds device count %" PRIhsz,
        physical_device_ordinal, context->physical_device_count);
  }
  if (IREE_UNLIKELY(!range.host_pointer || range.device_pointer == 0 ||
                    range.byte_length == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU source context loaded code-object range is empty");
  }
  if (IREE_UNLIKELY(!context->loaded_code_object_ranges)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU source context loaded code-object range table is unavailable");
  }
  context->loaded_code_object_ranges[physical_device_ordinal] = range;
  return iree_ok_status();
}

bool iree_hal_amdgpu_source_context_try_translate_device_span(
    const iree_hal_amdgpu_source_context_t* context,
    iree_host_size_t physical_device_ordinal, uint64_t device_pointer,
    uint64_t byte_length, iree_const_byte_span_t* out_host_span) {
  IREE_ASSERT_ARGUMENT(out_host_span);
  *out_host_span = iree_const_byte_span_empty();
  if (!context || physical_device_ordinal >= context->physical_device_count ||
      !context->loaded_code_object_ranges || byte_length > IREE_HOST_SIZE_MAX) {
    return false;
  }
  const iree_hal_amdgpu_loaded_code_object_range_t* range =
      &context->loaded_code_object_ranges[physical_device_ordinal];
  if (!range->host_pointer || device_pointer < range->device_pointer) {
    return false;
  }
  const uint64_t range_offset = device_pointer - range->device_pointer;
  if (range_offset > range->byte_length ||
      byte_length > range->byte_length - range_offset) {
    return false;
  }
  *out_host_span = iree_make_const_byte_span(range->host_pointer + range_offset,
                                             (iree_host_size_t)byte_length);
  return true;
}

iree_status_t iree_hal_amdgpu_source_context_set_sanitizer_site_table(
    iree_hal_amdgpu_source_context_t* context, iree_const_byte_span_t table) {
  IREE_ASSERT_ARGUMENT(context);
  if (IREE_UNLIKELY(iree_const_byte_span_is_empty(table))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table is empty");
  }
  if (IREE_UNLIKELY(table.data_length <
                    IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_LENGTH)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table is smaller than the "
        "header");
  }

  const uint32_t magic = iree_hal_amdgpu_source_context_load_u32(
      table.data, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_MAGIC_OFFSET);
  const uint8_t version = iree_hal_amdgpu_source_context_load_u8(
      table.data, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_VERSION_OFFSET);
  const uint8_t header_length = iree_hal_amdgpu_source_context_load_u8(
      table.data, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_HEADER_LENGTH_OFFSET);
  const uint16_t record_length = iree_hal_amdgpu_source_context_load_u16(
      table.data, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_RECORD_LENGTH_OFFSET);
  if (IREE_UNLIKELY(
          magic != IREE_HAL_AMDGPU_LOOM_SITE_TABLE_MAGIC ||
          version != IREE_HAL_AMDGPU_LOOM_SITE_TABLE_VERSION ||
          header_length != IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_LENGTH ||
          record_length != IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_LENGTH)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table has unsupported header: "
        "magic=0x%08x version=%u header_length=%u record_length=%u",
        magic, version, header_length, record_length);
  }

  iree_hal_amdgpu_source_context_site_table_t site_table;
  memset(&site_table, 0, sizeof(site_table));
  site_table.data = table.data;
  site_table.data_length = table.data_length;
  site_table.row_count = iree_hal_amdgpu_source_context_load_u32(
      table.data, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_ROW_COUNT_OFFSET);
  site_table.record_length = record_length;
  site_table.string_table_offset = iree_hal_amdgpu_source_context_load_u32(
      table.data,
      IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_STRING_TABLE_OFFSET_OFFSET);
  site_table.string_table_length = iree_hal_amdgpu_source_context_load_u32(
      table.data,
      IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_STRING_TABLE_LENGTH_OFFSET);
  site_table.payload_data_offset = iree_hal_amdgpu_source_context_load_u32(
      table.data,
      IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_PAYLOAD_DATA_OFFSET_OFFSET);
  site_table.payload_data_length = iree_hal_amdgpu_source_context_load_u32(
      table.data,
      IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_PAYLOAD_DATA_LENGTH_OFFSET);

  iree_host_size_t record_data_length = 0;
  if (!iree_host_size_checked_mul(site_table.row_count,
                                  site_table.record_length,
                                  &record_data_length) ||
      !iree_hal_amdgpu_source_context_span_in_bounds(
          IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_LENGTH, record_data_length,
          table.data_length) ||
      !iree_hal_amdgpu_source_context_span_in_bounds(
          site_table.string_table_offset, site_table.string_table_length,
          table.data_length) ||
      !iree_hal_amdgpu_source_context_span_in_bounds(
          site_table.payload_data_offset, site_table.payload_data_length,
          table.data_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source context sanitizer site table sections are out of range");
  }

  for (uint32_t i = 0; i < site_table.row_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_source_context_validate_sanitizer_site_table_record(
            &site_table, i));
  }

  context->sanitizer_site_table = site_table;
  iree_atomic_store(&context->sanitizer_site_table_published, 1,
                    iree_memory_order_release);
  return iree_ok_status();
}

bool iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
    const iree_hal_amdgpu_source_context_t* context, uint64_t site_id,
    iree_hal_device_event_site_t* out_site) {
  IREE_ASSERT_ARGUMENT(out_site);
  *out_site = iree_hal_device_event_site_default();
  if (!context || site_id > UINT32_MAX) return false;
  if (!iree_atomic_load(&context->sanitizer_site_table_published,
                        iree_memory_order_acquire)) {
    return false;
  }

  const iree_hal_amdgpu_source_context_site_table_t* table =
      &context->sanitizer_site_table;
  if (!table->data || site_id >= table->row_count) return false;

  const uint8_t* record = table->data +
                          IREE_HAL_AMDGPU_LOOM_SITE_TABLE_HEADER_LENGTH +
                          ((iree_host_size_t)site_id * table->record_length);
  const uint32_t flags = iree_hal_amdgpu_source_context_load_u32(
      record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_FLAGS_OFFSET);

  iree_hal_device_event_site_t site = iree_hal_device_event_site_default();
  site.site_id = site_id;
  site.operation_name = iree_hal_amdgpu_source_context_loom_op_kind_name(
      iree_hal_amdgpu_source_context_load_u32(
          record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_OP_KIND_OFFSET));
  if (iree_any_bit_set(
          flags, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION)) {
    const uint32_t source_name_offset = iree_hal_amdgpu_source_context_load_u32(
        record,
        IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET);
    const uint32_t source_name_length = iree_hal_amdgpu_source_context_load_u32(
        record,
        IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET);
    const char* source_name = (const char*)table->data +
                              table->string_table_offset + source_name_offset;
    site.source_file = iree_make_string_view(source_name, source_name_length);
    site.start_line = iree_hal_amdgpu_source_context_load_u32(
        record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_START_LINE_OFFSET);
    site.start_column = iree_hal_amdgpu_source_context_load_u32(
        record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_START_COLUMN_OFFSET);
    site.end_line = iree_hal_amdgpu_source_context_load_u32(
        record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_END_LINE_OFFSET);
    site.end_column = iree_hal_amdgpu_source_context_load_u32(
        record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_END_COLUMN_OFFSET);
  }
  if (iree_any_bit_set(flags,
                       IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_HAS_PAYLOAD)) {
    const uint32_t payload_offset = iree_hal_amdgpu_source_context_load_u32(
        record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_PAYLOAD_OFFSET_OFFSET);
    const uint32_t payload_length = iree_hal_amdgpu_source_context_load_u32(
        record, IREE_HAL_AMDGPU_LOOM_SITE_TABLE_RECORD_PAYLOAD_LENGTH_OFFSET);
    site.producer_payload = iree_make_const_byte_span(
        table->data + table->payload_data_offset + payload_offset,
        payload_length);
  }

  *out_site = site;
  return true;
}
