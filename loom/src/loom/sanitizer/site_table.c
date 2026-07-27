// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/site_table.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "loom/ir/module.h"

#define LOOM_SANITIZER_SITE_TABLE_LOCATION_MAX_DEPTH 64
#define LOOM_SANITIZER_SITE_TABLE_STRING_OFFSET_INVALID UINT32_MAX

typedef struct loom_sanitizer_site_table_file_location_t {
  // True when a file source range was found.
  bool has_file_location;
  // Module source table ID for the file source range.
  loom_source_id_t source_id;
  // One-based starting line.
  uint16_t start_line;
  // One-based starting column.
  uint16_t start_column;
  // One-based ending line.
  uint16_t end_line;
  // One-based ending column.
  uint16_t end_column;
} loom_sanitizer_site_table_file_location_t;

static void loom_sanitizer_site_table_write_u8(uint8_t* data,
                                               iree_host_size_t offset,
                                               uint8_t value) {
  data[offset] = value;
}

static void loom_sanitizer_site_table_write_u16(uint8_t* data,
                                                iree_host_size_t offset,
                                                uint16_t value) {
  iree_unaligned_store_le_u16(data + offset, value);
}

static void loom_sanitizer_site_table_write_u32(uint8_t* data,
                                                iree_host_size_t offset,
                                                uint32_t value) {
  iree_unaligned_store_le_u32(data + offset, value);
}

static iree_status_t loom_sanitizer_site_table_find_file_location(
    const loom_module_t* module, loom_location_id_t location_id, uint8_t depth,
    loom_sanitizer_site_table_file_location_t* out_file_location) {
  if (out_file_location->has_file_location ||
      location_id == LOOM_LOCATION_UNKNOWN) {
    return iree_ok_status();
  }
  if (depth >= LOOM_SANITIZER_SITE_TABLE_LOCATION_MAX_DEPTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sanitizer site table location tree exceeded maximum depth %u",
        (unsigned)LOOM_SANITIZER_SITE_TABLE_LOCATION_MAX_DEPTH);
  }
  if ((iree_host_size_t)location_id >= module->locations.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "sanitizer site table location id %u out of range (%" PRIhsz
        " locations)",
        location_id, module->locations.count);
  }

  const loom_location_entry_t* entry = &module->locations.entries[location_id];
  switch (loom_location_get_kind(*entry)) {
    case LOOM_LOCATION_NONE:
    case LOOM_LOCATION_OPAQUE:
      return iree_ok_status();

    case LOOM_LOCATION_FILE:
      if (entry->file.source_id == LOOM_SOURCE_ID_INVALID ||
          entry->file.source_id >= module->sources.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "sanitizer site table source id %u out of range (%" PRIhsz
            " sources)",
            (unsigned)entry->file.source_id, module->sources.count);
      }
      *out_file_location = (loom_sanitizer_site_table_file_location_t){
          .has_file_location = true,
          .source_id = entry->file.source_id,
          .start_line = entry->file.start_line,
          .start_column = entry->file.start_col,
          .end_line = entry->file.end_line,
          .end_column = entry->file.end_col,
      };
      return iree_ok_status();

    case LOOM_LOCATION_FUSED:
      if (entry->fused.count > 0 && entry->fused.children == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "sanitizer site table fused location %u has no children",
            location_id);
      }
      for (uint32_t i = 0;
           i < entry->fused.count && !out_file_location->has_file_location;
           ++i) {
        IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_find_file_location(
            module, entry->fused.children[i], (uint8_t)(depth + 1),
            out_file_location));
      }
      return iree_ok_status();

    case LOOM_LOCATION_TAGGED:
      return loom_sanitizer_site_table_find_file_location(
          module, entry->tagged.child, (uint8_t)(depth + 1), out_file_location);

    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "sanitizer site table location id %u has unsupported kind %u",
          location_id, (unsigned)entry->kind);
  }
}

static iree_status_t loom_sanitizer_site_table_row_file_location(
    const loom_module_t* module, const loom_sanitizer_site_row_t* row,
    loom_sanitizer_site_table_file_location_t* out_file_location) {
  *out_file_location = (loom_sanitizer_site_table_file_location_t){0};
  const loom_location_id_t location_id =
      row->source_location != LOOM_LOCATION_UNKNOWN ? row->source_location
                                                    : row->location;
  return loom_sanitizer_site_table_find_file_location(module, location_id, 0,
                                                      out_file_location);
}

static iree_status_t loom_sanitizer_site_table_record_source_string(
    const loom_module_t* module,
    const loom_sanitizer_site_table_file_location_t* file_location,
    uint32_t* source_string_offsets,
    iree_host_size_t* inout_string_table_length) {
  if (!file_location->has_file_location) return iree_ok_status();

  uint32_t* source_string_offset =
      &source_string_offsets[file_location->source_id];
  if (*source_string_offset !=
      LOOM_SANITIZER_SITE_TABLE_STRING_OFFSET_INVALID) {
    return iree_ok_status();
  }

  const iree_string_view_t source_name =
      module->sources.entries[file_location->source_id];
  if (source_name.size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table source name is too large");
  }
  if (*inout_string_table_length >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table string table is too large");
  }
  *source_string_offset = (uint32_t)*inout_string_table_length;

  iree_host_size_t next_length = 0;
  if (!iree_host_size_checked_add(*inout_string_table_length, source_name.size,
                                  &next_length) ||
      !iree_host_size_checked_add(next_length, 1, &next_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table string table overflow");
  }
  *inout_string_table_length = next_length;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_site_table_payload_length(
    const loom_sanitizer_site_payload_t* payload,
    iree_host_size_t* out_payload_length) {
  *out_payload_length = 0;
  if (!iree_host_size_checked_add(LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH,
                                  payload->extension_data.data_length,
                                  out_payload_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table payload overflow");
  }
  if (*out_payload_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table payload is too large");
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_site_table_measure(
    const loom_module_t* module,
    const loom_sanitizer_site_collection_t* collection,
    uint32_t* source_string_offsets, iree_host_size_t* out_string_table_length,
    iree_host_size_t* out_payload_data_length) {
  *out_string_table_length = 0;
  *out_payload_data_length = 0;
  for (iree_host_size_t i = 0; i < collection->row_count; ++i) {
    const loom_sanitizer_site_row_t* row = &collection->rows[i];

    loom_sanitizer_site_table_file_location_t file_location = {0};
    IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_row_file_location(
        module, row, &file_location));
    IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_record_source_string(
        module, &file_location, source_string_offsets,
        out_string_table_length));

    if (loom_sanitizer_site_row_has_payload(row)) {
      iree_host_size_t payload_length = 0;
      IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_payload_length(
          &row->payload, &payload_length));
      if (!iree_host_size_checked_add(*out_payload_data_length, payload_length,
                                      out_payload_data_length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "sanitizer site table payload data overflow");
      }
    }
  }
  if (*out_string_table_length > UINT32_MAX ||
      *out_payload_data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table section is too large");
  }
  return iree_ok_status();
}

static void loom_sanitizer_site_table_write_source_strings(
    const loom_module_t* module, const uint32_t* source_string_offsets,
    uint8_t* string_table_data) {
  for (iree_host_size_t i = 0; i < module->sources.count; ++i) {
    const uint32_t string_offset = source_string_offsets[i];
    if (string_offset == LOOM_SANITIZER_SITE_TABLE_STRING_OFFSET_INVALID) {
      continue;
    }
    const iree_string_view_t source_name = module->sources.entries[i];
    uint8_t* target = string_table_data + string_offset;
    if (source_name.size > 0) {
      memcpy(target, source_name.data, source_name.size);
    }
    target[source_name.size] = 0;
  }
}

static iree_status_t loom_sanitizer_site_table_write_record(
    const loom_module_t* module, const loom_sanitizer_site_row_t* row,
    const uint32_t* source_string_offsets, uint8_t* record_data,
    uint8_t* payload_data, iree_host_size_t* inout_payload_data_length) {
  loom_sanitizer_site_table_record_flags_t record_flags = 0;
  iree_host_size_t payload_offset = 0;
  iree_host_size_t payload_length = 0;
  if (loom_sanitizer_site_row_has_payload(row)) {
    IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_payload_length(
        &row->payload, &payload_length));
    payload_offset = *inout_payload_data_length;
    iree_host_size_t encoded_length = 0;
    IREE_RETURN_IF_ERROR(loom_sanitizer_site_payload_encode(
        &row->payload,
        iree_make_byte_span(payload_data + payload_offset, payload_length),
        &encoded_length));
    if (encoded_length != payload_length) {
      IREE_ASSERT_UNREACHABLE(
          "sanitizer site table payload length changed during encoding");
      IREE_BUILTIN_UNREACHABLE();
    }
    *inout_payload_data_length += payload_length;
    record_flags |= LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_PAYLOAD;
  }

  uint32_t source_name_offset = 0;
  uint32_t source_name_length = 0;
  uint32_t start_line = 0;
  uint32_t start_column = 0;
  uint32_t end_line = 0;
  uint32_t end_column = 0;
  loom_sanitizer_site_table_source_kind_t source_kind =
      LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_NONE;
  loom_sanitizer_site_table_file_location_t file_location = {0};
  IREE_RETURN_IF_ERROR(
      loom_sanitizer_site_table_row_file_location(module, row, &file_location));
  if (file_location.has_file_location) {
    const iree_string_view_t source_name =
        module->sources.entries[file_location.source_id];
    source_name_offset = source_string_offsets[file_location.source_id];
    source_name_length = (uint32_t)source_name.size;
    start_line = file_location.start_line;
    start_column = file_location.start_column;
    end_line = file_location.end_line;
    end_column = file_location.end_column;
    source_kind = LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_FILE;
    record_flags |= LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION;
  }

  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET,
      row->site_id);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET,
      row->op_kind);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET, record_flags);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_PAYLOAD_OFFSET_OFFSET,
      (uint32_t)payload_offset);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_PAYLOAD_LENGTH_OFFSET,
      (uint32_t)payload_length);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET,
      source_name_offset);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET,
      source_name_length);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_START_LINE_OFFSET,
      start_line);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_START_COLUMN_OFFSET,
      start_column);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_END_LINE_OFFSET, end_line);
  loom_sanitizer_site_table_write_u32(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_END_COLUMN_OFFSET,
      end_column);
  loom_sanitizer_site_table_write_u16(
      record_data, LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET,
      source_kind);
  return iree_ok_status();
}

iree_status_t loom_sanitizer_site_table_encode(
    const loom_module_t* module,
    const loom_sanitizer_site_collection_t* collection,
    iree_arena_allocator_t* arena, iree_const_byte_span_t* out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  *out_table = iree_const_byte_span_empty();
  if (collection->row_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sanitizer site table has too many rows");
  }

  uint32_t* source_string_offsets = NULL;
  if (module->sources.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->sources.count, sizeof(*source_string_offsets),
        (void**)&source_string_offsets));
    for (iree_host_size_t i = 0; i < module->sources.count; ++i) {
      source_string_offsets[i] =
          LOOM_SANITIZER_SITE_TABLE_STRING_OFFSET_INVALID;
    }
  }

  iree_host_size_t string_table_length = 0;
  iree_host_size_t payload_data_length = 0;
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_measure(
      module, collection, source_string_offsets, &string_table_length,
      &payload_data_length));

  iree_host_size_t record_data_length = 0;
  if (!iree_host_size_checked_mul(collection->row_count,
                                  LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH,
                                  &record_data_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table record data overflow");
  }
  iree_host_size_t string_table_offset = 0;
  iree_host_size_t payload_data_offset = 0;
  iree_host_size_t total_length = 0;
  if (!iree_host_size_checked_add(LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH,
                                  record_data_length, &string_table_offset) ||
      !iree_host_size_checked_add(string_table_offset, string_table_length,
                                  &payload_data_offset) ||
      !iree_host_size_checked_add(payload_data_offset, payload_data_length,
                                  &total_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table size overflow");
  }
  if (string_table_offset > UINT32_MAX || string_table_length > UINT32_MAX ||
      payload_data_offset > UINT32_MAX || payload_data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sanitizer site table section offset is too large");
  }

  uint8_t* table_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, total_length, (void**)&table_data));
  memset(table_data, 0, total_length);

  loom_sanitizer_site_table_write_u32(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_MAGIC_OFFSET,
      LOOM_SANITIZER_SITE_TABLE_MAGIC);
  loom_sanitizer_site_table_write_u8(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_VERSION_OFFSET,
      LOOM_SANITIZER_SITE_TABLE_VERSION);
  loom_sanitizer_site_table_write_u8(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_HEADER_LENGTH_OFFSET,
      LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH);
  loom_sanitizer_site_table_write_u16(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_RECORD_LENGTH_OFFSET,
      LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  loom_sanitizer_site_table_write_u32(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_ROW_COUNT_OFFSET,
      (uint32_t)collection->row_count);
  loom_sanitizer_site_table_write_u32(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_OFFSET_OFFSET,
      (uint32_t)string_table_offset);
  loom_sanitizer_site_table_write_u32(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_LENGTH_OFFSET,
      (uint32_t)string_table_length);
  loom_sanitizer_site_table_write_u32(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_PAYLOAD_DATA_OFFSET_OFFSET,
      (uint32_t)payload_data_offset);
  loom_sanitizer_site_table_write_u32(
      table_data, LOOM_SANITIZER_SITE_TABLE_HEADER_PAYLOAD_DATA_LENGTH_OFFSET,
      (uint32_t)payload_data_length);

  if (source_string_offsets != NULL) {
    loom_sanitizer_site_table_write_source_strings(
        module, source_string_offsets, table_data + string_table_offset);
  }

  iree_host_size_t current_payload_data_length = 0;
  for (iree_host_size_t i = 0; i < collection->row_count; ++i) {
    uint8_t* record_data = table_data +
                           LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH +
                           (i * LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
    IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_write_record(
        module, &collection->rows[i], source_string_offsets, record_data,
        table_data + payload_data_offset, &current_payload_data_length));
  }
  if (current_payload_data_length != payload_data_length) {
    IREE_ASSERT_UNREACHABLE(
        "sanitizer site table payload data length changed during encoding");
    IREE_BUILTIN_UNREACHABLE();
  }

  *out_table = iree_make_const_byte_span(table_data, total_length);
  return iree_ok_status();
}
