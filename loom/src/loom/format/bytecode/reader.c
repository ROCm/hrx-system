// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/diagnostic.h"
#include "loom/format/bytecode/varint.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"
#include "loom/verify/verify.h"

// Keep reader allocation guards aligned with the bytecode format comment.
#define LOOM_BYTECODE_MAX_SECTION_COUNT 256
#define LOOM_BYTECODE_MAX_STRING_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_STRING_LENGTH (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_TYPE_COUNT (UINT64_C(1) << 16)
#define LOOM_BYTECODE_MAX_OP_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_SYMBOL_COUNT (UINT64_C(1) << 16)
#define LOOM_BYTECODE_MAX_LOCATION_COUNT (UINT64_C(1) << 24)
#define LOOM_BYTECODE_MAX_ENCODING_COUNT (UINT64_C(1) << 16)
#define LOOM_BYTECODE_MAX_REGION_DEPTH 256

enum {
  LOOM_BYTECODE_ATTR_I64 = 0,
  LOOM_BYTECODE_ATTR_F64 = 1,
  LOOM_BYTECODE_ATTR_STRING = 2,
  LOOM_BYTECODE_ATTR_BOOL = 3,
  LOOM_BYTECODE_ATTR_ENUM = 4,
  LOOM_BYTECODE_ATTR_I64_ARRAY = 5,
  LOOM_BYTECODE_ATTR_SYMBOL = 6,
  LOOM_BYTECODE_ATTR_TYPE = 7,
  LOOM_BYTECODE_ATTR_PREDICATE_LIST = 8,
  LOOM_BYTECODE_ATTR_DICT = 9,
  LOOM_BYTECODE_ATTR_ENCODING = 10,
  LOOM_BYTECODE_ATTR_BYTES = 11,
  LOOM_BYTECODE_ATTR_COUNT,
};

typedef struct loom_bytecode_reader_cursor_t {
  loom_bytecode_cursor_t cursor;  // Bounded cursor over the current byte range.
  uint64_t absolute_offset;       // Absolute file offset for cursor byte 0.
  iree_string_view_t range_name;  // Human-readable range name for diagnostics.
} loom_bytecode_reader_cursor_t;

typedef struct loom_bytecode_reader_module_t {
  uint32_t name_offset;     // Offset into the file string pool.
  uint16_t name_length;     // Module name byte length.
  uint16_t flags;           // Module flags from the directory entry.
  uint64_t offset;          // Absolute module byte offset.
  uint64_t length;          // Module byte length.
  iree_string_view_t name;  // Name view into the file string pool.
} loom_bytecode_reader_module_t;

typedef struct loom_bytecode_reader_section_t {
  uint16_t kind;                 // Wire section kind.
  uint16_t flags;                // Section flags.
  uint64_t offset;               // Module-relative byte offset.
  uint64_t length;               // Section byte length.
  uint64_t absolute_offset;      // Absolute file byte offset.
  iree_const_byte_span_t bytes;  // Section payload bytes.
} loom_bytecode_reader_section_t;

typedef struct loom_bytecode_reader_state_t {
  iree_const_byte_span_t bytecode;  // Full bytecode file bytes.
  iree_string_view_t filename;      // Logical input name for diagnostics.
  loom_context_t* context;          // Dialect and encoding registry context.
  iree_arena_allocator_t* arena;    // Transient metadata arena.
  // Persistent arena for public metadata index output.
  iree_arena_allocator_t* metadata_arena;
  // Diagnostic context used by bytecode/diagnostic helpers.
  loom_bytecode_reader_diagnostic_context_t diagnostic_context;
  loom_bytecode_read_result_t result;  // Public result accumulator.
  uint8_t format_version;              // File header bytecode format version.
  iree_string_view_t producer;         // File header producer string.
  // Optional public metadata index populated while validating bytecode.
  loom_bytecode_file_metadata_t* output_metadata;
  // Optional public metadata record for the module currently being decoded.
  loom_bytecode_module_metadata_t* current_module_metadata;
  // Allocation and table-count summary for the current module.
  loom_bytecode_module_metadata_summary_t current_module_summary;

  iree_string_view_t file_string_pool;     // File-level module-name pool.
  loom_bytecode_reader_module_t* modules;  // Module directory entries.
  iree_host_size_t module_count;           // Number of module entries.

  iree_string_view_t* strings;    // Current module STRINGS entries.
  iree_host_size_t string_count;  // Number of current module strings.
  iree_string_view_t* sources;    // Current module SOURCES entries.
  loom_source_id_t* source_ids;   // Bytecode source index to module source ID.
  iree_host_size_t source_count;  // Number of current module sources.
  loom_type_t* types;             // Current module TYPES entries.
  iree_host_size_t type_count;    // Number of current module types.
  const loom_op_vtable_t** ops;   // Current module OPS resolved vtables.
  loom_op_kind_t* op_kinds;       // Current module OPS resolved op kinds.
  iree_host_size_t op_count;      // Number of current module OPS entries.
  // Current module ENCODINGS family vtables.
  const loom_encoding_vtable_t** encoding_families;
  loom_string_id_t* encoding_family_name_ids;  // Family name string IDs.
  iree_host_size_t encoding_family_count;      // Number of encoding families.
  iree_host_size_t encoding_count;             // Number of encoding instances.
  iree_host_size_t location_count;             // Number of location entries.
  iree_host_size_t symbol_count;               // Number of symbol entries.
  iree_arena_block_pool_t* block_pool;         // Arena block source.
  iree_allocator_t host_allocator;  // Host allocator for output module.
  loom_module_t* output_module;     // Module being materialized.
} loom_bytecode_reader_state_t;

typedef struct loom_bytecode_body_counts_t {
  uint64_t value_count;   // SSA values defined while decoding a body.
  uint64_t region_count;  // Regions decoded, including nested regions.
  uint64_t block_count;   // Blocks decoded, including nested regions.
  uint64_t op_count;      // Operations decoded, including nested regions.
} loom_bytecode_body_counts_t;

typedef struct loom_bytecode_body_reader_t {
  loom_bytecode_reader_state_t* reader;  // Owning file reader.
  iree_arena_allocator_t* arena;         // Per-function scratch arena.
  iree_string_view_t symbol_name;        // Function name for diagnostics.
  uint64_t body_offset;                  // Absolute byte offset of the body.
  loom_value_id_t* value_map;            // Function-local value number map.
  uint64_t value_capacity;               // Expected value count from summary.
  uint64_t next_value_number;            // Next value definition to decode.
  uint64_t available_value_count;        // Reserved prefix available to types.
  // Values already defined by the containing symbol signature.
  const loom_value_id_t* predefined_values;
  // Number of leading value definitions that must reuse predefined values.
  uint16_t predefined_value_count;
  loom_bytecode_body_counts_t counts;  // Actual decoded body counts.
} loom_bytecode_body_reader_t;

static void loom_bytecode_reader_cursor_initialize(
    const uint8_t* data, iree_host_size_t length, uint64_t absolute_offset,
    iree_string_view_t range_name, loom_bytecode_reader_cursor_t* out_cursor) {
  loom_bytecode_cursor_initialize(data, length, &out_cursor->cursor);
  out_cursor->absolute_offset = absolute_offset;
  out_cursor->range_name = range_name;
}

static uint64_t loom_bytecode_reader_cursor_absolute_position(
    const loom_bytecode_reader_cursor_t* cursor) {
  return cursor->absolute_offset + (uint64_t)cursor->cursor.position;
}

static const char* loom_bytecode_section_name(uint16_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_SECTION_STRINGS:
      return "STRINGS";
    case LOOM_BYTECODE_SECTION_SOURCES:
      return "SOURCES";
    case LOOM_BYTECODE_SECTION_TYPES:
      return "TYPES";
    case LOOM_BYTECODE_SECTION_ENCODINGS:
      return "ENCODINGS";
    case LOOM_BYTECODE_SECTION_OPS:
      return "OPS";
    case LOOM_BYTECODE_SECTION_LOCATIONS:
      return "LOCATIONS";
    case LOOM_BYTECODE_SECTION_SYMBOLS:
      return "SYMBOLS";
    case LOOM_BYTECODE_SECTION_IR:
      return "IR";
    case LOOM_BYTECODE_SECTION_RESOURCES:
      return "RESOURCES";
    default:
      return "UNKNOWN";
  }
}

static bool loom_bytecode_reader_has_errors(
    const loom_bytecode_reader_state_t* reader) {
  return reader->result.error_count > 0;
}

static iree_status_t loom_bytecode_reader_emit(
    loom_bytecode_reader_state_t* reader, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    uint64_t offset, uint64_t length) {
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++reader->result.error_count;
  } else if (error->severity == LOOM_DIAGNOSTIC_WARNING) {
    ++reader->result.warning_count;
  }
  return loom_bytecode_reader_emit_diagnostic(
      &reader->diagnostic_context, error, params, param_count,
      loom_bytecode_reader_byte_range(offset, length));
}

static iree_status_t loom_bytecode_reader_emit_unexpected_end(
    loom_bytecode_reader_state_t* reader, uint64_t offset, uint64_t needed,
    uint64_t available) {
  loom_diagnostic_param_t params[] = {
      loom_param_u64(offset),
      loom_param_u64(needed),
      loom_param_u64(available),
  };
  return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_003, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_emit_invalid_field(
    loom_bytecode_reader_state_t* reader, iree_string_view_t section_name,
    iree_string_view_t table_name, uint64_t record_index,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t failure_code) {
  ++reader->result.error_count;
  return loom_bytecode_reader_emit_invalid_record_field(
      &reader->diagnostic_context, section_name, table_name, record_index,
      field_name, offset, failure_code);
}

static iree_status_t loom_bytecode_reader_emit_range_error(
    loom_bytecode_reader_state_t* reader, iree_string_view_t range_name,
    uint64_t offset, uint64_t length, uint64_t container_length) {
  ++reader->result.error_count;
  return loom_bytecode_reader_emit_invalid_range(&reader->diagnostic_context,
                                                 range_name, offset, length,
                                                 container_length);
}

static iree_status_t loom_bytecode_reader_emit_count_exceeds(
    loom_bytecode_reader_state_t* reader, iree_string_view_t table_name,
    uint64_t count, uint64_t limit, uint64_t offset) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(table_name),
      loom_param_u64(count),
      loom_param_u64(limit),
  };
  return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_009, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_emit_invalid_ir_body(
    loom_bytecode_body_reader_t* body_reader, uint64_t offset,
    iree_string_view_t failure_code) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(body_reader->symbol_name),
      loom_param_u64(offset),
      loom_param_string(failure_code),
  };
  return loom_bytecode_reader_emit(body_reader->reader, LOOM_ERR_BYTECODE_016,
                                   params, IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_body_reader_lookup_value(
    loom_bytecode_body_reader_t* body_reader, uint64_t value_number,
    uint64_t value_limit, uint64_t offset, iree_string_view_t failure_code,
    loom_value_id_t* out_value_id) {
  if (value_number >= value_limit) {
    return loom_bytecode_reader_emit_invalid_ir_body(body_reader, offset,
                                                     failure_code);
  }
  *out_value_id = body_reader->value_map[value_number];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_emit_table_ref(
    loom_bytecode_reader_state_t* reader, iree_string_view_t table_name,
    uint64_t ref_id, uint64_t table_count, uint64_t offset) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(table_name),
      loom_param_u64(ref_id),
      loom_param_u64(table_count),
  };
  return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_012, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_emit_enum_value(
    loom_bytecode_reader_state_t* reader, iree_string_view_t field_name,
    uint64_t actual_value, uint64_t case_count, uint64_t offset) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u64(actual_value),
      loom_param_u64(case_count),
  };
  return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_011, params,
                                   IREE_ARRAYSIZE(params), offset, 1);
}

static iree_status_t loom_bytecode_reader_read_u8(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint8_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 1)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 1,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u8(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_u16_le(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint16_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 2)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 2,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u16_le(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_u32_le(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint32_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 4)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 4,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u32_le(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_u64_le(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t* out_value) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 8)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), 8,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u64_le(&cursor->cursor, out_value);
}

static iree_status_t loom_bytecode_reader_read_uvarint(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t* out_value) {
  uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  iree_status_t status = loom_uvarint_decode(&cursor->cursor, out_value);
  if (iree_status_is_ok(status)) return iree_ok_status();

  bool reached_end = iree_status_is_out_of_range(status);
  iree_status_free(status);
  iree_string_view_t failure_code =
      reached_end ? IREE_SV("unterminated_varint")
                  : IREE_SV("noncanonical_or_uint64_overflow");
  loom_diagnostic_param_t params[] = {
      loom_param_u64(offset),
      loom_param_string(failure_code),
  };
  return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_008, params,
                                   IREE_ARRAYSIZE(params), offset, 0);
}

static iree_status_t loom_bytecode_reader_read_svarint(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    int64_t* out_value) {
  uint64_t zigzag = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &zigzag));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  *out_value = (int64_t)((zigzag >> 1) ^ -(zigzag & 1));
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_span(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t length, iree_const_byte_span_t* out_span) {
  if (length > (uint64_t)IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_range_error(
        reader, cursor->range_name,
        loom_bytecode_reader_cursor_absolute_position(cursor), length,
        cursor->absolute_offset + cursor->cursor.length);
  }
  iree_host_size_t host_length = (iree_host_size_t)length;
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, host_length)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, loom_bytecode_reader_cursor_absolute_position(cursor), length,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_span(&cursor->cursor, host_length, out_span);
}

static iree_status_t loom_bytecode_reader_read_comment_list(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    iree_arena_allocator_t* arena, const iree_string_view_t** out_comments,
    iree_host_size_t* out_comment_count) {
  if (out_comments) *out_comments = NULL;
  if (out_comment_count) *out_comment_count = 0;

  uint64_t count_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > UINT16_MAX || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, cursor->range_name, IREE_SV("comment_list"), 0,
        IREE_SV("comment_count"), count_offset,
        IREE_SV("comment_count_exceeds_field_width"));
  }

  iree_string_view_t* comments = NULL;
  if (out_comments && count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, (iree_host_size_t)count, sizeof(iree_string_view_t),
        (void**)&comments));
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t length_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t length = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (length > LOOM_BYTECODE_MAX_STRING_LENGTH) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, cursor->range_name, IREE_SV("comment_list"), i,
          IREE_SV("comment_length"), length_offset,
          IREE_SV("comment_length_exceeds_maximum"));
    }
    iree_const_byte_span_t span = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_span(reader, cursor, length, &span));
    if (comments) {
      comments[i] =
          iree_make_string_view((const char*)span.data, span.data_length);
    }
  }
  if (out_comments) *out_comments = comments;
  if (out_comment_count) *out_comment_count = (iree_host_size_t)count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_expect_empty(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    iree_string_view_t table_name) {
  if (loom_bytecode_cursor_is_empty(&cursor->cursor)) return iree_ok_status();
  return loom_bytecode_reader_emit_invalid_field(
      reader, cursor->range_name, table_name, 0, IREE_SV("trailing_bytes"),
      loom_bytecode_reader_cursor_absolute_position(cursor),
      IREE_SV("section_has_unread_trailing_bytes"));
}

static bool loom_bytecode_reader_string_is_valid_utf8(iree_string_view_t text) {
  return iree_unicode_utf8_validate(text);
}

static iree_status_t loom_bytecode_reader_validate_string_ref(
    loom_bytecode_reader_state_t* reader, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  if (string_id >= reader->string_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(reader->string_count),
    };
    return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_010, params,
                                     IREE_ARRAYSIZE(params), offset, 0);
  }
  *out_string = reader->strings[string_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_type_ref_bounded(
    loom_bytecode_reader_state_t* reader, uint64_t type_id, uint64_t type_count,
    iree_string_view_t field_name, uint64_t offset, loom_type_t* out_type) {
  (void)field_name;
  if (type_id >= type_count) {
    return loom_bytecode_reader_emit_table_ref(reader, IREE_SV("TYPES"),
                                               type_id, type_count, offset);
  }
  *out_type = reader->types[type_id];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_type_ref(
    loom_bytecode_reader_state_t* reader, uint64_t type_id,
    iree_string_view_t field_name, uint64_t offset, loom_type_t* out_type) {
  return loom_bytecode_reader_validate_type_ref_bounded(
      reader, type_id, reader->type_count, field_name, offset, out_type);
}

static iree_status_t loom_bytecode_reader_validate_encoding_ref(
    loom_bytecode_reader_state_t* reader, uint64_t encoding_id,
    uint64_t offset) {
  if (encoding_id == 0 || encoding_id > reader->encoding_count) {
    return loom_bytecode_reader_emit_table_ref(reader, IREE_SV("ENCODINGS"),
                                               encoding_id,
                                               reader->encoding_count, offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_location_ref(
    loom_bytecode_reader_state_t* reader, uint64_t location_id,
    uint64_t offset) {
  if (location_id >= reader->location_count) {
    return loom_bytecode_reader_emit_table_ref(reader, IREE_SV("LOCATIONS"),
                                               location_id,
                                               reader->location_count, offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_op_ref(
    loom_bytecode_reader_state_t* reader, uint64_t op_table_index_plus1,
    uint64_t offset, const loom_op_vtable_t** out_vtable) {
  if (op_table_index_plus1 == 0 || op_table_index_plus1 > reader->op_count) {
    return loom_bytecode_reader_emit_table_ref(
        reader, IREE_SV("OPS"), op_table_index_plus1, reader->op_count, offset);
  }
  *out_vtable = reader->ops[op_table_index_plus1 - 1];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_range(
    loom_bytecode_reader_state_t* reader, iree_string_view_t range_name,
    uint64_t offset, uint64_t length, uint64_t container_length) {
  if (length > UINT64_MAX - offset || offset + length > container_length) {
    return loom_bytecode_reader_emit_range_error(reader, range_name, offset,
                                                 length, container_length);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_string_table(
    loom_bytecode_reader_state_t* reader,
    loom_bytecode_reader_section_t section, iree_string_view_t table_name,
    uint64_t count_limit, iree_string_view_t** out_strings,
    iree_host_size_t* out_count) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section.bytes.data, section.bytes.data_length, section.absolute_offset,
      table_name, &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > count_limit || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(reader, table_name, count,
                                                   count_limit, count_offset);
  }

  iree_string_view_t* strings = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)count, sizeof(iree_string_view_t),
        (void**)&strings));
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t string_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t length = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (length > LOOM_BYTECODE_MAX_STRING_LENGTH) {
      return loom_bytecode_reader_emit_count_exceeds(
          reader, IREE_SV("string_length"), length,
          LOOM_BYTECODE_MAX_STRING_LENGTH, string_offset);
    }
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_span(reader, &cursor, length, &bytes));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t text =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!loom_bytecode_reader_string_is_valid_utf8(text)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, table_name, IREE_SV("string"), i, IREE_SV("utf8_data"),
          string_offset, IREE_SV("string_payload_is_not_valid_utf_8"));
    }
    strings[i] = text;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_expect_empty(reader, &cursor, table_name));
  *out_strings = strings;
  *out_count = (iree_host_size_t)count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_strings(
    loom_bytecode_reader_state_t* reader) {
  for (iree_host_size_t i = 0; i < reader->string_count; ++i) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        reader->output_module, reader->strings[i], &string_id));
    if (string_id != i) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("STRINGS"), IREE_SV("string"), i, IREE_SV("string"),
          0,
          IREE_SV("string_table_must_be_deduplicated_and_preserve_intern_ids"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_sources(
    loom_bytecode_reader_state_t* reader) {
  if (reader->source_count > LOOM_SOURCE_ID_INVALID) {
    return loom_bytecode_reader_emit_count_exceeds(reader, IREE_SV("SOURCES"),
                                                   reader->source_count,
                                                   LOOM_SOURCE_ID_INVALID, 0);
  }
  if (reader->source_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      reader->arena, reader->source_count, sizeof(loom_source_id_t),
      (void**)&reader->source_ids));
  for (iree_host_size_t i = 0; i < reader->source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_register_source(
        reader->output_module, reader->sources[i], &reader->source_ids[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_types(
    loom_bytecode_reader_state_t* reader) {
  for (iree_host_size_t i = 0; i < reader->type_count; ++i) {
    loom_type_id_t type_id = 0;
    IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
        reader->output_module, reader->types[i], &type_id));
    if (type_id != i) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("TYPES"), IREE_SV("type"), i, IREE_SV("type"), 0,
          IREE_SV("type_table_must_be_deduplicated_and_topologically_ordered"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_predicate_list(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    bool value_args_are_strings) {
  uint64_t predicate_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &predicate_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  for (uint64_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    uint8_t predicate_kind = 0;
    uint8_t arg_count = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &predicate_kind));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (predicate_kind >= LOOM_PREDICATE_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("predicate_kind"), predicate_kind,
          LOOM_PREDICATE_COUNT_, kind_offset);
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &arg_count));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    for (uint8_t arg_index = 0; arg_index < arg_count; ++arg_index) {
      uint8_t tag = 0;
      uint64_t tag_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &tag));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (tag == 1) {
        uint64_t encoded_value = 0;
        uint64_t value_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, cursor, &encoded_value));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (value_args_are_strings) {
          iree_string_view_t unused = iree_string_view_empty();
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
              reader, encoded_value, IREE_SV("predicate_value_name"),
              value_offset, &unused));
        }
      } else if (tag == 2) {
        int64_t value = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_svarint(reader, cursor, &value));
        (void)value;
      } else {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("predicate_arg_tag"), tag, 3, tag_offset);
      }
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_attr_value(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint8_t kind, bool predicate_value_args_are_strings) {
  switch (kind) {
    case LOOM_BYTECODE_ATTR_I64: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_svarint(reader, cursor, &value));
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_F64: {
      uint64_t bits = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u64_le(reader, cursor, &bits));
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_STRING: {
      uint64_t string_id = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &string_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t unused = iree_string_view_empty();
      return loom_bytecode_reader_validate_string_ref(
          reader, string_id, IREE_SV("attribute_string"), offset, &unused);
    }
    case LOOM_BYTECODE_ATTR_ENUM: {
      uint8_t unused = 0;
      return loom_bytecode_reader_read_u8(reader, cursor, &unused);
    }
    case LOOM_BYTECODE_ATTR_BOOL: {
      uint8_t value = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, cursor, &value));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (value > 1) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("bool_attribute"), value, 2, offset);
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_I64_ARRAY: {
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &count));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      for (uint64_t i = 0; i < count; ++i) {
        int64_t value = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_svarint(reader, cursor, &value));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BYTES: {
      uint64_t byte_length = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &byte_length));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_const_byte_span_t unused = iree_const_byte_span_empty();
      return loom_bytecode_reader_read_span(reader, cursor, byte_length,
                                            &unused);
    }
    case LOOM_BYTECODE_ATTR_SYMBOL: {
      uint64_t symbol_name_id = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &symbol_name_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t unused = iree_string_view_empty();
      return loom_bytecode_reader_validate_string_ref(
          reader, symbol_name_id, IREE_SV("attribute_symbol"), offset, &unused);
    }
    case LOOM_BYTECODE_ATTR_TYPE: {
      uint64_t type_id = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &type_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      loom_type_t unused = {0};
      return loom_bytecode_reader_validate_type_ref(
          reader, type_id, IREE_SV("attribute_type"), offset, &unused);
    }
    case LOOM_BYTECODE_ATTR_PREDICATE_LIST:
      return loom_bytecode_reader_skip_predicate_list(
          reader, cursor, predicate_value_args_are_strings);
    case LOOM_BYTECODE_ATTR_DICT: {
      uint64_t entry_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &entry_count));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t previous_key = iree_string_view_empty();
      for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
        uint64_t key_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t key_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        iree_string_view_t key = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, key_id, IREE_SV("dict_key"), key_offset, &key));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (entry_index > 0 &&
            iree_string_view_compare(key, previous_key) <= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, cursor->range_name, IREE_SV("dict"), entry_index,
              IREE_SV("key_id"), key_offset,
              IREE_SV("dictionary_keys_are_not_in_canonical_order"));
        }
        previous_key = key;
        uint8_t value_kind = 0;
        uint64_t value_kind_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("attribute_kind"), value_kind,
              LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_attr_value(
            reader, cursor, value_kind, predicate_value_args_are_strings));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENCODING: {
      uint64_t encoding_id = 0;
      uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &encoding_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      return loom_bytecode_reader_validate_encoding_ref(reader, encoding_id,
                                                        offset);
    }
    default:
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), kind, LOOM_BYTECODE_ATTR_COUNT,
          loom_bytecode_reader_cursor_absolute_position(cursor));
  }
}

static iree_status_t loom_bytecode_reader_read_predicate_list_attr(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    loom_bytecode_body_reader_t* body_reader, loom_attribute_t* out_attr) {
  uint64_t predicate_count = 0;
  uint64_t count_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &predicate_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (predicate_count > UINT16_MAX || predicate_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("predicate_list"), predicate_count, UINT16_MAX,
        count_offset);
  }
  loom_predicate_t* predicates = NULL;
  if (predicate_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &reader->output_module->arena, (iree_host_size_t)predicate_count,
        sizeof(loom_predicate_t), (void**)&predicates));
    memset(predicates, 0,
           (iree_host_size_t)predicate_count * sizeof(loom_predicate_t));
  }
  for (uint64_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    loom_predicate_t* predicate = &predicates[predicate_index];
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &predicate->kind));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (predicate->kind >= LOOM_PREDICATE_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("predicate_kind"), predicate->kind,
          LOOM_PREDICATE_COUNT_, kind_offset);
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &predicate->arg_count));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    uint8_t expected_arg_count =
        loom_predicate_kind_argument_count(predicate->kind);
    if (predicate->arg_count != expected_arg_count ||
        predicate->arg_count > IREE_ARRAYSIZE(predicate->args)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, cursor->range_name, IREE_SV("predicate"), predicate_index,
          IREE_SV("arg_count"), kind_offset + 1,
          IREE_SV("predicate_arity_does_not_match_its_kind"));
    }
    for (uint8_t arg_index = 0; arg_index < predicate->arg_count; ++arg_index) {
      uint64_t tag_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(
          reader, cursor, &predicate->arg_tags[arg_index]));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      switch (predicate->arg_tags[arg_index]) {
        case LOOM_PRED_ARG_VALUE: {
          uint64_t value_offset =
              loom_bytecode_reader_cursor_absolute_position(cursor);
          uint64_t encoded_value = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              reader, cursor, &encoded_value));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          if (body_reader) {
            loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
            IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_lookup_value(
                body_reader, encoded_value, body_reader->next_value_number,
                value_offset,
                IREE_SV("predicate value reference must target a previously "
                        "defined value"),
                &value_id));
            if (loom_bytecode_reader_has_errors(reader))
              return iree_ok_status();
            predicate->args[arg_index] = (int64_t)value_id;
          } else {
            iree_string_view_t unused = iree_string_view_empty();
            IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
                reader, encoded_value, IREE_SV("predicate_value_name"),
                value_offset, &unused));
            if (loom_bytecode_reader_has_errors(reader))
              return iree_ok_status();
            predicate->args[arg_index] = (int64_t)encoded_value;
          }
          break;
        }
        case LOOM_PRED_ARG_CONST: {
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_svarint(
              reader, cursor, &predicate->args[arg_index]));
          break;
        }
        default:
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("predicate_arg_tag"),
              predicate->arg_tags[arg_index], LOOM_PRED_ARG_COUNT_, tag_offset);
      }
    }
  }
  *out_attr = loom_attr_predicate_list(predicates, (uint16_t)predicate_count);
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_attr_value(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    loom_bytecode_body_reader_t* body_reader,
    const loom_attr_descriptor_t* descriptor, uint8_t kind,
    loom_attribute_t* out_attr) {
  (void)descriptor;
  switch (kind) {
    case LOOM_BYTECODE_ATTR_I64: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_svarint(reader, cursor, &value));
      *out_attr = loom_attr_i64(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_F64: {
      uint64_t bits = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u64_le(reader, cursor, &bits));
      double value = 0.0;
      memcpy(&value, &bits, sizeof(value));
      *out_attr = loom_attr_f64(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_STRING: {
      uint64_t string_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t string_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &string_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t unused = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, string_id, IREE_SV("attribute_string"), string_offset,
          &unused));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      *out_attr = loom_attr_string((loom_string_id_t)string_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BOOL: {
      uint64_t value_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, cursor, &value));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (value > 1) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("bool_attribute"), value, 2, value_offset);
      }
      *out_attr = loom_attr_bool(value != 0);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENUM: {
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, cursor, &value));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      *out_attr = loom_attr_enum(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_I64_ARRAY: {
      uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &count));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (count > UINT16_MAX || count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            reader, IREE_SV("i64_array"), count, UINT16_MAX, count_offset);
      }
      int64_t* values = NULL;
      if (count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &reader->output_module->arena, (iree_host_size_t)count,
            sizeof(int64_t), (void**)&values));
      }
      for (uint64_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_svarint(reader, cursor, &values[i]));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      *out_attr = loom_attr_i64_array(values, (uint16_t)count);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BYTES: {
      uint64_t byte_length_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t byte_length = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &byte_length));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (byte_length > UINT32_MAX || byte_length > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            reader, IREE_SV("bytes_attribute"), byte_length, UINT32_MAX,
            byte_length_offset);
      }
      iree_const_byte_span_t span = iree_const_byte_span_empty();
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_span(reader, cursor, byte_length, &span));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      uint8_t* bytes = NULL;
      if (span.data_length > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            &reader->output_module->arena, span.data_length, (void**)&bytes));
        memcpy(bytes, span.data, span.data_length);
      }
      *out_attr = loom_attr_bytes(bytes, (uint32_t)span.data_length);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SYMBOL: {
      uint64_t name_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t name_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &name_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t unused = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, name_id, IREE_SV("attribute_symbol"), name_offset, &unused));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      uint16_t symbol_id = loom_module_find_symbol(reader->output_module,
                                                   (loom_string_id_t)name_id);
      if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("symbol"), name_offset,
            IREE_SV("symbol_attribute_references_an_unknown_symbol"));
      }
      *out_attr = loom_attr_symbol((loom_symbol_ref_t){0, symbol_id});
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_TYPE: {
      uint64_t type_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t type_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &type_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      loom_type_t unused = {0};
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref(
          reader, type_id, IREE_SV("attribute_type"), type_offset, &unused));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      *out_attr = loom_attr_type((loom_type_id_t)type_id);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_PREDICATE_LIST:
      return loom_bytecode_reader_read_predicate_list_attr(
          reader, cursor, body_reader, out_attr);
    case LOOM_BYTECODE_ATTR_DICT: {
      uint64_t count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t entry_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &entry_count));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (entry_count > UINT16_MAX || entry_count > IREE_HOST_SIZE_MAX) {
        return loom_bytecode_reader_emit_count_exceeds(
            reader, IREE_SV("dict_entries"), entry_count, UINT16_MAX,
            count_offset);
      }
      loom_named_attr_t* entries = NULL;
      if (entry_count > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            reader->arena, (iree_host_size_t)entry_count,
            sizeof(loom_named_attr_t), (void**)&entries));
      }
      iree_string_view_t previous_key = iree_string_view_empty();
      for (uint64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
        uint64_t key_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        uint64_t key_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        iree_string_view_t key = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, key_id, IREE_SV("dict_key"), key_offset, &key));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (entry_index > 0 &&
            iree_string_view_compare(key, previous_key) <= 0) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, cursor->range_name, IREE_SV("dict"), entry_index,
              IREE_SV("key_id"), key_offset,
              IREE_SV("dictionary_keys_are_not_in_canonical_order"));
        }
        previous_key = key;
        uint8_t value_kind = 0;
        uint64_t value_kind_offset =
            loom_bytecode_reader_cursor_absolute_position(cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("attribute_kind"), value_kind,
              LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
        }
        entries[entry_index].name_id = (loom_string_id_t)key_id;
        entries[entry_index].reserved = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_attr_value(
            reader, cursor, body_reader, NULL, value_kind,
            &entries[entry_index].value));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      return loom_module_make_canonical_attr_dict(
          reader->output_module,
          loom_make_named_attr_slice(entries, (iree_host_size_t)entry_count),
          out_attr);
    }
    case LOOM_BYTECODE_ATTR_ENCODING: {
      uint64_t encoding_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &encoding_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_encoding_ref(
          reader, encoding_id, encoding_offset));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (encoding_id > reader->output_module->encodings.count) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, cursor->range_name, IREE_SV("attribute"), 0,
            IREE_SV("encoding"), encoding_offset,
            IREE_SV("encoding_attribute_references_an_unavailable_encoding"));
      }
      *out_attr = loom_attr_encoding((uint16_t)encoding_id);
      return iree_ok_status();
    }
    default:
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), kind, LOOM_BYTECODE_ATTR_COUNT,
          loom_bytecode_reader_cursor_absolute_position(cursor));
  }
}

static iree_status_t loom_bytecode_reader_read_encodings(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("ENCODINGS"), &cursor);

  uint64_t family_count = 0;
  uint64_t family_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &family_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (family_count > LOOM_BYTECODE_MAX_ENCODING_COUNT ||
      family_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("encoding_families"), family_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, family_count_offset);
  }
  if (family_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, (iree_host_size_t)family_count,
                                  sizeof(const loom_encoding_vtable_t*),
                                  (void**)&reader->encoding_families));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)family_count, sizeof(loom_string_id_t),
        (void**)&reader->encoding_family_name_ids));
  }
  reader->encoding_family_count = (iree_host_size_t)family_count;
  for (uint64_t i = 0; i < family_count; ++i) {
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("encoding_family_name"), name_offset, &name));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    const loom_encoding_vtable_t* vtable =
        loom_context_lookup_encoding_vtable(reader->context, name);
    if (!vtable) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("ENCODINGS"), IREE_SV("family"), i,
          IREE_SV("name_id"), name_offset,
          IREE_SV("encoding_family_is_not_registered_in_the_context"));
    }
    reader->encoding_families[i] = vtable;
    reader->encoding_family_name_ids[i] = (loom_string_id_t)name_id;
  }

  uint64_t instance_count = 0;
  uint64_t instance_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &instance_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (instance_count > LOOM_BYTECODE_MAX_ENCODING_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("encoding_instances"), instance_count,
        LOOM_BYTECODE_MAX_ENCODING_COUNT, instance_count_offset);
  }
  reader->encoding_count = (iree_host_size_t)instance_count;
  for (uint64_t i = 0; i < instance_count; ++i) {
    uint64_t family_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t family_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &family_index));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (family_index >= family_count) {
      return loom_bytecode_reader_emit_table_ref(
          reader, IREE_SV("encoding_families"), family_index, family_count,
          family_offset);
    }
    uint64_t alias_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t alias_plus_one = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &alias_plus_one));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (alias_plus_one > reader->string_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("alias_string_id_plus1")),
          loom_param_u64(alias_plus_one - 1),
          loom_param_u64(reader->string_count),
      };
      return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_010, params,
                                       IREE_ARRAYSIZE(params), alias_offset, 0);
    }
    uint64_t param_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &param_count));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t previous_param = iree_string_view_empty();
    for (uint64_t param_index = 0; param_index < param_count; ++param_index) {
      uint64_t name_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t name_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t param_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, name_id, IREE_SV("encoding_param_name"), name_offset,
          &param_name));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (param_index > 0 &&
          iree_string_view_compare(param_name, previous_param) <= 0) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("ENCODINGS"), IREE_SV("instance"), i,
            IREE_SV("param_name"), name_offset,
            IREE_SV("encoding_parameters_are_not_in_canonical_order"));
      }
      previous_param = param_name;
      uint8_t value_kind = 0;
      uint64_t kind_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &value_kind));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("attribute_kind"), value_kind,
            LOOM_BYTECODE_ATTR_COUNT, kind_offset);
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_attr_value(
          reader, &cursor, value_kind, true));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    }
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor,
                                           IREE_SV("ENCODINGS"));
}

static iree_status_t loom_bytecode_reader_materialize_encodings(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("ENCODINGS"), &cursor);

  uint64_t family_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &family_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  for (uint64_t i = 0; i < family_count; ++i) {
    uint64_t unused_name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &unused_name_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  uint64_t instance_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &instance_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  for (uint64_t instance_index = 0; instance_index < instance_count;
       ++instance_index) {
    uint64_t family_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t family_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &family_index));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (family_index >= reader->encoding_family_count) {
      return loom_bytecode_reader_emit_table_ref(
          reader, IREE_SV("encoding_families"), family_index,
          reader->encoding_family_count, family_offset);
    }

    uint64_t alias_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t alias_plus_one = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &alias_plus_one));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (alias_plus_one > reader->string_count) {
      return loom_bytecode_reader_emit_table_ref(
          reader, IREE_SV("STRINGS"), alias_plus_one - 1, reader->string_count,
          alias_offset);
    }

    uint64_t param_count = 0;
    uint64_t param_count_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &param_count));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (param_count > UINT8_MAX || param_count > IREE_HOST_SIZE_MAX) {
      return loom_bytecode_reader_emit_count_exceeds(
          reader, IREE_SV("encoding_params"), param_count, UINT8_MAX,
          param_count_offset);
    }

    loom_named_attr_t* params = NULL;
    if (param_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->arena, (iree_host_size_t)param_count,
          sizeof(loom_named_attr_t), (void**)&params));
    }
    for (uint64_t param_index = 0; param_index < param_count; ++param_index) {
      uint64_t name_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t name_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      iree_string_view_t unused_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, name_id, IREE_SV("encoding_param_name"), name_offset,
          &unused_name));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      uint8_t value_kind = 0;
      uint64_t value_kind_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &value_kind));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("attribute_kind"), value_kind,
            LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
      }
      params[param_index].name_id = (loom_string_id_t)name_id;
      params[param_index].reserved = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_attr_value(
          reader, &cursor, NULL, NULL, value_kind, &params[param_index].value));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    }

    loom_encoding_t encoding = {
        .name_id = reader->encoding_family_name_ids[family_index],
        .alias_id = alias_plus_one == 0
                        ? LOOM_STRING_ID_INVALID
                        : (loom_string_id_t)(alias_plus_one - 1),
        .attribute_count = (uint8_t)param_count,
        .attributes = params,
    };
    uint16_t encoding_id = 0;
    IREE_RETURN_IF_ERROR(loom_module_add_encoding(reader->output_module,
                                                  &encoding, &encoding_id));
    if (encoding_id != instance_index + 1) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("ENCODINGS"), IREE_SV("instance"), instance_index,
          IREE_SV("encoding"), family_offset,
          IREE_SV("encoding_table_must_be_deduplicated_in_canonical_order"));
    }
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor,
                                           IREE_SV("ENCODINGS"));
}

static iree_status_t loom_bytecode_reader_decode_type_kind(
    loom_bytecode_reader_state_t* reader, uint8_t kind_byte, uint64_t offset,
    loom_type_kind_t* out_kind) {
  switch (kind_byte) {
    case LOOM_BYTECODE_TYPE_NONE:
      *out_kind = LOOM_TYPE_NONE;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_SCALAR:
      *out_kind = LOOM_TYPE_SCALAR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_TILE:
      *out_kind = LOOM_TYPE_TILE;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_TENSOR:
      *out_kind = LOOM_TYPE_TENSOR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_GROUP:
      *out_kind = LOOM_TYPE_GROUP;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_FUNCTION:
      *out_kind = LOOM_TYPE_FUNCTION;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_DIALECT:
      *out_kind = LOOM_TYPE_DIALECT;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_ENCODING:
      *out_kind = LOOM_TYPE_ENCODING;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_POOL:
      *out_kind = LOOM_TYPE_POOL;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_VECTOR:
      *out_kind = LOOM_TYPE_VECTOR;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_VIEW:
      *out_kind = LOOM_TYPE_VIEW;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_BUFFER:
      *out_kind = LOOM_TYPE_BUFFER;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_REGISTER:
      *out_kind = LOOM_TYPE_REGISTER;
      return iree_ok_status();
    case LOOM_BYTECODE_TYPE_STORAGE:
      *out_kind = LOOM_TYPE_STORAGE;
      return iree_ok_status();
    default: {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(kind_byte),
          loom_param_u64(offset),
      };
      return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_004, params,
                                       IREE_ARRAYSIZE(params), offset, 1);
    }
  }
}

static iree_status_t loom_bytecode_reader_build_shaped_type(
    loom_bytecode_reader_state_t* reader, loom_type_kind_t kind,
    loom_scalar_type_t element_type, uint8_t rank, uint8_t attachment,
    uint64_t encoding_instance, const uint64_t* dims, loom_type_t* out_type,
    uint64_t offset) {
  if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
    return loom_bytecode_reader_emit_enum_value(
        reader, IREE_SV("element_type"), element_type, LOOM_SCALAR_TYPE_COUNT_,
        offset);
  }
  if (rank > LOOM_TYPE_MAX_RANK) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("TYPES"), IREE_SV("type"), 0, IREE_SV("rank"), offset,
        IREE_SV("rank_exceeds_loom_type_maximum"));
  }
  if (kind == LOOM_TYPE_VECTOR && rank == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("TYPES"), IREE_SV("type"), 0, IREE_SV("rank"), offset,
        IREE_SV("vector_rank_zero"));
  }

  uint16_t encoding_id = 0;
  loom_encoding_flags_t encoding_flags = 0;
  switch (attachment) {
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE:
      if (encoding_instance != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("none_encoding_attachment_must_have_id_0"));
      }
      break;
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC: {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_encoding_ref(
          reader, encoding_instance, offset));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (encoding_instance > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("encoding_id_exceeds_runtime_type_field_width"));
      }
      encoding_id = (uint16_t)encoding_instance;
      break;
    }
    case LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA:
      if (encoding_instance != 0) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
            IREE_SV("encoding_instance"), offset,
            IREE_SV("dynamic_encoding_attachment_must_have_id_0"));
      }
      encoding_flags = LOOM_ENCODING_FLAG_SSA;
      break;
    default:
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("encoding_attachment"), attachment,
          LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA + 1, offset);
  }
  if (kind == LOOM_TYPE_VECTOR && (encoding_id != 0 || encoding_flags != 0)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("TYPES"), IREE_SV("type"), 0,
        IREE_SV("encoding_attachment"), offset,
        IREE_SV("vector_types_must_not_carry_encoding_or_layout_attachments"));
  }

  loom_type_t type = {0};
  if (rank == 0) {
    type = loom_type_shaped_0d(kind, element_type, encoding_id);
  } else if (rank == 1) {
    type = loom_type_shaped_1d(kind, element_type, dims[0], encoding_id);
  } else if (rank == 2) {
    type =
        loom_type_shaped_2d(kind, element_type, dims[0], dims[1], encoding_id);
  } else {
    loom_overflow_dim_t* overflow_dims = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(reader->arena, rank,
                                                   sizeof(loom_overflow_dim_t),
                                                   (void**)&overflow_dims));
    bool all_static = true;
    for (uint8_t i = 0; i < rank; ++i) {
      overflow_dims[i] = dims[i];
      if (loom_dim_is_dynamic(dims[i])) all_static = false;
    }
    uint8_t flags = all_static ? LOOM_TYPE_FLAG_ALL_STATIC : 0;
    type.header = loom_type_make_header(kind, element_type, rank, flags);
    type.encoding_id = encoding_id;
    type.dims[0] = (uint64_t)(uintptr_t)overflow_dims;
  }
  type.encoding_flags = encoding_flags;
  *out_type = type;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_types(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("TYPES"), &cursor);

  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_TYPE_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("TYPES"), count, LOOM_BYTECODE_MAX_TYPE_COUNT,
        count_offset);
  }
  if (count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, (iree_host_size_t)count,
                                  sizeof(loom_type_t), (void**)&reader->types));
  }
  reader->type_count = (iree_host_size_t)count;
  for (uint64_t type_index = 0; type_index < count; ++type_index) {
    uint64_t type_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, &cursor, &kind_byte));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    loom_type_kind_t kind = LOOM_TYPE_NONE;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_decode_type_kind(
        reader, kind_byte, type_offset, &kind));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

    loom_type_t type = {0};
    switch (kind) {
      case LOOM_TYPE_NONE:
        type = loom_type_none();
        break;
      case LOOM_TYPE_SCALAR: {
        uint8_t element_type = 0;
        uint64_t element_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &element_type));
        if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("scalar_type"), element_type,
              LOOM_SCALAR_TYPE_COUNT_, element_offset);
        }
        type = loom_type_scalar((loom_scalar_type_t)element_type);
        break;
      }
      case LOOM_TYPE_TILE:
      case LOOM_TYPE_TENSOR:
      case LOOM_TYPE_VECTOR:
      case LOOM_TYPE_VIEW: {
        uint8_t element_type = 0;
        uint8_t rank = 0;
        uint8_t attachment = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &element_type));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &rank));
        uint64_t attachment_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &attachment));
        uint64_t encoding_instance = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            reader, &cursor, &encoding_instance));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        uint64_t dims[LOOM_TYPE_MAX_RANK] = {0};
        if (rank > LOOM_TYPE_MAX_RANK) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("rank"), type_offset,
              IREE_SV("rank_exceeds_loom_type_maximum"));
        }
        for (uint8_t i = 0; i < rank; ++i) {
          uint8_t is_dynamic = 0;
          uint64_t dim_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_u8(reader, &cursor, &is_dynamic));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          if (is_dynamic == 0) {
            uint64_t size = 0;
            IREE_RETURN_IF_ERROR(
                loom_bytecode_reader_read_uvarint(reader, &cursor, &size));
            if (size > LOOM_DIM_MAX_STATIC_SIZE) {
              return loom_bytecode_reader_emit_invalid_field(
                  reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
                  IREE_SV("dim_size"), dim_offset,
                  IREE_SV("static_dimension_exceeds_loom_maximum"));
            }
            dims[i] = loom_dim_pack_static((int64_t)size);
          } else if (is_dynamic == 1) {
            dims[i] = loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID);
          } else {
            return loom_bytecode_reader_emit_enum_value(
                reader, IREE_SV("is_dynamic"), is_dynamic, 2, dim_offset);
          }
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_build_shaped_type(
            reader, kind, (loom_scalar_type_t)element_type, rank, attachment,
            encoding_instance, dims, &type, attachment_offset));
        break;
      }
      case LOOM_TYPE_GROUP: {
        uint8_t scope = 0;
        uint64_t scope_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &scope));
        if (scope >= LOOM_GROUP_SCOPE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("group_scope"), scope, LOOM_GROUP_SCOPE_COUNT_,
              scope_offset);
        }
        type.header = loom_type_make_raw_header(
            LOOM_TYPE_GROUP, scope, 0,
            LOOM_TYPE_FLAG_INLINE_DIMS | LOOM_TYPE_FLAG_ALL_STATIC);
        break;
      }
      case LOOM_TYPE_FUNCTION: {
        uint64_t arg_count = 0;
        uint64_t result_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &arg_count));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &result_count));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (arg_count > UINT16_MAX || result_count > UINT16_MAX ||
            arg_count + result_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("signature_count"), type_offset,
              IREE_SV("function_signature_exceeds_runtime_field_width"));
        }
        iree_host_size_t total_count =
            (iree_host_size_t)(arg_count + result_count);
        iree_host_size_t total_size =
            sizeof(loom_func_type_data_t) + total_count * sizeof(loom_type_t);
        loom_func_type_data_t* data = NULL;
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(reader->arena, total_size, (void**)&data));
        data->arg_count = (uint16_t)arg_count;
        data->result_count = (uint16_t)result_count;
        data->reserved = 0;
        for (iree_host_size_t i = 0; i < total_count; ++i) {
          uint64_t ref_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t type_id = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &type_id));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref_bounded(
              reader, type_id, type_index, IREE_SV("signature_type"),
              ref_offset, &data->types[i]));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        }
        type = loom_type_function(data);
        break;
      }
      case LOOM_TYPE_DIALECT: {
        uint64_t name_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t name_id = 0;
        uint64_t param_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &param_count));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        iree_string_view_t unused_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, name_id, IREE_SV("dialect_type_name"), name_offset,
            &unused_name));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (param_count > UINT16_MAX || param_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              reader, IREE_SV("dialect_type_params"), param_count, UINT16_MAX,
              type_offset);
        }
        loom_type_t* params = NULL;
        if (param_count > 0) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              reader->arena, (iree_host_size_t)param_count, sizeof(loom_type_t),
              (void**)&params));
        }
        for (uint64_t i = 0; i < param_count; ++i) {
          uint64_t ref_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t type_id = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &type_id));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref_bounded(
              reader, type_id, type_index, IREE_SV("dialect_type_param"),
              ref_offset, &params[i]));
        }
        type = param_count == 0
                   ? loom_type_dialect_opaque((loom_string_id_t)name_id)
                   : loom_type_dialect((loom_string_id_t)name_id,
                                       (uint16_t)param_count, params);
        break;
      }
      case LOOM_TYPE_REGISTER: {
        uint64_t payload0_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t payload0 = 0;
        uint64_t payload1 = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &payload0));
        uint64_t payload1_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &payload1));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (payload0 == 0) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("register_payload0"), payload0_offset,
              IREE_SV("register_descriptor_set_stable_id_must_be_non_zero"));
        }
        if (((payload1 >> 16) & UINT32_MAX) == 0) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
              IREE_SV("register_payload1"), payload1_offset,
              IREE_SV("register_unit_count_must_be_non_zero"));
        }
        type = loom_type_register_payload(payload0, payload1);
        break;
      }
      case LOOM_TYPE_STORAGE: {
        uint8_t space = 0;
        uint64_t space_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &space));
        if (space >= LOOM_STORAGE_SPACE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("storage_space"), space,
              LOOM_STORAGE_SPACE_COUNT_, space_offset);
        }
        type = loom_type_storage((loom_storage_space_t)space);
        break;
      }
      case LOOM_TYPE_ENCODING: {
        uint8_t role = 0;
        uint64_t role_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &role));
        if (role >= LOOM_ENCODING_ROLE_COUNT_) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("encoding_role"), role, LOOM_ENCODING_ROLE_COUNT_,
              role_offset);
        }
        type = loom_type_encoding_with_role((loom_encoding_role_t)role);
        break;
      }
      case LOOM_TYPE_POOL: {
        uint8_t is_dynamic = 0;
        uint64_t dim_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &is_dynamic));
        if (is_dynamic == 0) {
          uint64_t size = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &size));
          if (size > LOOM_DIM_MAX_STATIC_SIZE) {
            return loom_bytecode_reader_emit_invalid_field(
                reader, IREE_SV("TYPES"), IREE_SV("type"), type_index,
                IREE_SV("block_size"), dim_offset,
                IREE_SV("static_pool_block_size_exceeds_loom_maximum"));
          }
          type = loom_type_pool(loom_dim_pack_static((int64_t)size));
        } else if (is_dynamic == 1) {
          type = loom_type_pool(loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID));
        } else {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("is_dynamic"), is_dynamic, 2, dim_offset);
        }
        break;
      }
      case LOOM_TYPE_BUFFER:
        type = loom_type_buffer();
        break;
      default:
        break;
    }
    reader->types[type_index] = type;
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("TYPES"));
}

static iree_status_t loom_bytecode_reader_read_ops(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("OPS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_OP_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("OPS"), count, LOOM_BYTECODE_MAX_OP_COUNT,
        count_offset);
  }
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)count, sizeof(const loom_op_vtable_t*),
        (void**)&reader->ops));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)count, sizeof(loom_op_kind_t),
        (void**)&reader->op_kinds));
  }
  reader->op_count = (iree_host_size_t)count;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t op_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("op_name"), name_offset, &op_name));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
    const loom_op_vtable_t* vtable =
        loom_context_lookup_op_by_name(reader->context, op_name, &kind);
    if (!vtable) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("OPS"), IREE_SV("op"), i, IREE_SV("name_id"),
          name_offset, IREE_SV("op_name_is_not_registered_in_the_context"));
    }
    reader->ops[i] = vtable;
    reader->op_kinds[i] = kind;
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("OPS"));
}

static iree_status_t loom_bytecode_reader_read_locations(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("LOCATIONS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_LOCATION_COUNT || count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("LOCATIONS"), count, LOOM_BYTECODE_MAX_LOCATION_COUNT,
        count_offset);
  }
  reader->location_count = (iree_host_size_t)count;
  for (uint64_t i = 0; i < count; ++i) {
    uint8_t kind = 0;
    uint8_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &flags));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("flags"), kind_offset + 1,
          IREE_SV("location_has_unsupported_flag_bits"));
    }
    switch ((loom_location_kind_t)kind) {
      case LOOM_LOCATION_NONE:
        break;
      case LOOM_LOCATION_FILE: {
        uint64_t source_id = 0;
        uint64_t source_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &source_id));
        if (source_id >= reader->source_count) {
          return loom_bytecode_reader_emit_table_ref(
              reader, IREE_SV("SOURCES"), source_id, reader->source_count,
              source_offset);
        }
        for (int field = 0; field < 4; ++field) {
          uint64_t value = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &value));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        }
        break;
      }
      case LOOM_LOCATION_FUSED: {
        uint64_t child_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &child_count));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        for (uint64_t child_index = 0; child_index < child_count;
             ++child_index) {
          uint64_t child_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t child = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &child));
          if (child >= i) {
            return loom_bytecode_reader_emit_table_ref(
                reader, IREE_SV("LOCATIONS"), child, i, child_offset);
          }
        }
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        uint64_t source_id = 0;
        uint64_t source_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &source_id));
        if (source_id >= reader->source_count) {
          return loom_bytecode_reader_emit_table_ref(
              reader, IREE_SV("SOURCES"), source_id, reader->source_count,
              source_offset);
        }
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &data_length));
        iree_const_byte_span_t unused = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            reader, &cursor, data_length, &unused));
        break;
      }
      case LOOM_LOCATION_TAGGED: {
        uint64_t tag = 0;
        uint64_t tag_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &tag));
        if (tag == LOOM_LOCATION_TAG_INVALID || tag > UINT16_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
              IREE_SV("tag"), tag_offset,
              IREE_SV("tagged location tag must be in [1, 65535]"));
        }
        uint64_t child = 0;
        uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &child));
        if (child >= i) {
          return loom_bytecode_reader_emit_table_ref(
              reader, IREE_SV("LOCATIONS"), child, i, child_offset);
        }
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &data_length));
        iree_const_byte_span_t unused = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            reader, &cursor, data_length, &unused));
        break;
      }
      default:
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("location_kind"), kind, LOOM_LOCATION_COUNT_,
            kind_offset);
    }
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor,
                                           IREE_SV("LOCATIONS"));
}

static iree_status_t loom_bytecode_reader_read_location_coordinate(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t location_index, iree_string_view_t field_name,
    uint16_t* out_value) {
  uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &value));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (value > UINT16_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("LOCATIONS"), IREE_SV("location"), location_index,
        field_name, offset,
        IREE_SV("file_location_coordinate_exceeds_runtime_field_width"));
  }
  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_source_ref(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    loom_source_id_t* out_source_id) {
  uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t source_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &source_index));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (source_index >= reader->source_count || !reader->source_ids) {
    return loom_bytecode_reader_emit_table_ref(
        reader, IREE_SV("SOURCES"), source_index, reader->source_count, offset);
  }
  *out_source_id = reader->source_ids[source_index];
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_locations(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      section->bytes.data, section->bytes.data_length, section->absolute_offset,
      IREE_SV("LOCATIONS"), &cursor);

  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count != reader->location_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("LOCATIONS"), IREE_SV("location"), 0, IREE_SV("count"),
        count_offset,
        IREE_SV("location_count_changed_between_validation_and_materialize"));
  }

  for (uint64_t i = 0; i < count; ++i) {
    uint8_t kind = 0;
    uint8_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &flags));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (flags & ~LOOM_LOCATION_FLAG_SYNTHETIC) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("flags"), kind_offset + 1,
          IREE_SV("location_has_unsupported_flag_bits"));
    }

    loom_location_entry_t entry = {
        .kind = (loom_location_kind_t)kind,
        .flags = flags,
    };
    switch (kind) {
      case LOOM_LOCATION_NONE:
        if (i != 0 || flags != 0) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
              IREE_SV("kind"), kind_offset,
              IREE_SV("only_location_0_may_be_the_unflagged_none_location"));
        }
        continue;
      case LOOM_LOCATION_FILE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_source_ref(
            reader, &cursor, &entry.file.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_location_coordinate(
            reader, &cursor, i, IREE_SV("start_line"), &entry.file.start_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_location_coordinate(
            reader, &cursor, i, IREE_SV("start_col"), &entry.file.start_col));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_location_coordinate(
            reader, &cursor, i, IREE_SV("end_line"), &entry.file.end_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_location_coordinate(
            reader, &cursor, i, IREE_SV("end_col"), &entry.file.end_col));
        break;
      }
      case LOOM_LOCATION_FUSED: {
        uint64_t child_count_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t child_count = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &child_count));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (child_count > UINT32_MAX || child_count > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              reader, IREE_SV("location_children"), child_count, UINT32_MAX,
              child_count_offset);
        }
        loom_location_id_t* children = NULL;
        if (child_count > 0) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &reader->output_module->arena, (iree_host_size_t)child_count,
              sizeof(loom_location_id_t), (void**)&children));
        }
        for (uint64_t child_index = 0; child_index < child_count;
             ++child_index) {
          uint64_t child_offset =
              loom_bytecode_reader_cursor_absolute_position(&cursor);
          uint64_t child = 0;
          IREE_RETURN_IF_ERROR(
              loom_bytecode_reader_read_uvarint(reader, &cursor, &child));
          if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
          if (child >= i) {
            return loom_bytecode_reader_emit_table_ref(
                reader, IREE_SV("LOCATIONS"), child, i, child_offset);
          }
          children[child_index] = (loom_location_id_t)child;
        }
        entry.fused.count = (uint32_t)child_count;
        entry.fused.children = children;
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_source_ref(
            reader, &cursor, &entry.opaque.source_id));
        uint64_t data_length_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &data_length));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              reader, IREE_SV("opaque_location_data"), data_length, UINT32_MAX,
              data_length_offset);
        }
        iree_const_byte_span_t data_span = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            reader, &cursor, data_length, &data_span));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        uint8_t* data = NULL;
        if (data_span.data_length > 0) {
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(&reader->output_module->arena,
                                  data_span.data_length, (void**)&data));
          memcpy(data, data_span.data, data_span.data_length);
        }
        entry.opaque.data_length = (uint32_t)data_span.data_length;
        entry.opaque.data = data;
        break;
      }
      case LOOM_LOCATION_TAGGED: {
        uint64_t tag_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t tag = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &tag));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (tag == LOOM_LOCATION_TAG_INVALID || tag > UINT16_MAX) {
          return loom_bytecode_reader_emit_invalid_field(
              reader, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
              IREE_SV("tag"), tag_offset,
              IREE_SV("tagged location tag must be in [1, 65535]"));
        }
        uint64_t child_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t child = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &child));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (child >= i) {
          return loom_bytecode_reader_emit_table_ref(
              reader, IREE_SV("LOCATIONS"), child, i, child_offset);
        }
        uint64_t data_length_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t data_length = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &data_length));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (data_length > UINT32_MAX || data_length > IREE_HOST_SIZE_MAX) {
          return loom_bytecode_reader_emit_count_exceeds(
              reader, IREE_SV("tagged_location_data"), data_length, UINT32_MAX,
              data_length_offset);
        }
        iree_const_byte_span_t data_span = iree_const_byte_span_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
            reader, &cursor, data_length, &data_span));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        uint8_t* data = NULL;
        if (data_span.data_length > 0) {
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(&reader->output_module->arena,
                                  data_span.data_length, (void**)&data));
          memcpy(data, data_span.data, data_span.data_length);
        }
        entry.tagged.tag = (loom_location_tag_t)tag;
        entry.tagged.child = (loom_location_id_t)child;
        entry.tagged.data_length = (uint32_t)data_span.data_length;
        entry.tagged.data = data;
        break;
      }
      default:
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("location_kind"), kind, LOOM_LOCATION_COUNT_,
            kind_offset);
    }
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(
        loom_module_add_location(reader->output_module, entry, &location_id));
    if (location_id != i) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("LOCATIONS"), IREE_SV("location"), i,
          IREE_SV("location_id"), kind_offset,
          IREE_SV("location_table_must_preserve_bytecode_location_ids"));
    }
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor,
                                           IREE_SV("LOCATIONS"));
}

static iree_status_t loom_bytecode_body_reader_read_region(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    loom_op_t* parent_op, uint32_t depth, loom_region_t** out_region);

static iree_status_t loom_bytecode_body_reader_bind_type(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_type_t base_type,
    uint64_t dim_binding_count, loom_type_t* out_type) {
  loom_type_t type = base_type;
  uint8_t rank = loom_type_rank(base_type);
  uint64_t dynamic_count = 0;
  uint64_t dims[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t i = 0; i < rank; ++i) {
    dims[i] = loom_type_dim(base_type, i);
    if (loom_dim_is_dynamic(dims[i])) ++dynamic_count;
  }
  if (dim_binding_count != dynamic_count) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("dynamic_dimension_binding_count_does_not_match_the_type"));
  }
  for (uint8_t i = 0; i < rank; ++i) {
    if (!loom_dim_is_dynamic(dims[i])) continue;
    int64_t value_number = 0;
    uint64_t ref_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_svarint(
        body_reader->reader, cursor, &value_number));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    if (value_number < 0) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, ref_offset,
          IREE_SV("dynamic_dimension_value_reference_must_be_non_negative"));
    }
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_lookup_value(
        body_reader, (uint64_t)value_number, body_reader->available_value_count,
        ref_offset,
        IREE_SV("dynamic dimension value reference must target an available "
                "value"),
        &value_id));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    dims[i] = loom_dim_pack_dynamic(value_id);
  }

  uint64_t encoding_binding = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &encoding_binding));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();

  if (loom_type_has_ssa_encoding(base_type)) {
    if (encoding_binding == 0) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, loom_bytecode_reader_cursor_absolute_position(cursor),
          IREE_SV("ssa_encoding_binding_is_required_by_the_type"));
    }
    uint64_t value_number = encoding_binding - 1;
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_lookup_value(
        body_reader, value_number, body_reader->available_value_count,
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("ssa_encoding_value_reference_must_target_an_available_value"),
        &value_id));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    if (value_id > UINT16_MAX) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, loom_bytecode_reader_cursor_absolute_position(cursor),
          IREE_SV("ssa_encoding_value_id_exceeds_type_payload_width"));
    }
    type.encoding_id = (uint16_t)value_id;
  } else if (encoding_binding != 0) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("ssa_encoding_binding_is_present_for_a_type_without_one"));
  }
  uint16_t rebound_encoding_id = type.encoding_id;

  if (rank == 0) {
    *out_type = type;
    return iree_ok_status();
  }
  if (rank == 1) {
    type = loom_type_shaped_1d(loom_type_kind(base_type),
                               loom_type_element_type(base_type), dims[0],
                               rebound_encoding_id);
  } else if (rank == 2) {
    type = loom_type_shaped_2d(loom_type_kind(base_type),
                               loom_type_element_type(base_type), dims[0],
                               dims[1], rebound_encoding_id);
  } else {
    loom_overflow_dim_t* overflow_dims = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(body_reader->arena, rank,
                                                   sizeof(loom_overflow_dim_t),
                                                   (void**)&overflow_dims));
    uint8_t flags = 0;
    bool all_static = true;
    for (uint8_t i = 0; i < rank; ++i) {
      overflow_dims[i] = dims[i];
      if (loom_dim_is_dynamic(dims[i])) all_static = false;
    }
    if (all_static) flags |= LOOM_TYPE_FLAG_ALL_STATIC;
    type.header =
        loom_type_make_header(loom_type_kind(base_type),
                              loom_type_element_type(base_type), rank, flags);
    type.dims[0] = (uint64_t)(uintptr_t)overflow_dims;
  }
  type.encoding_flags = base_type.encoding_flags;
  type.encoding_id = rebound_encoding_id;
  *out_type = type;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_reader_define_value(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_value_id_t* out_value_id) {
  uint64_t name_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t name_id = 0;
  uint64_t type_id = 0;
  uint64_t dim_binding_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(body_reader->reader, cursor, &name_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(body_reader->reader, cursor, &type_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &dim_binding_count));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();

  if (body_reader->next_value_number >= body_reader->available_value_count) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, name_offset,
        IREE_SV("value_definition_was_not_reserved_before_decoding"));
  }
  if (name_id != 0) {
    iree_string_view_t unused_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        body_reader->reader, name_id, IREE_SV("value_name"), name_offset,
        &unused_name));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
  }
  uint64_t type_offset = name_offset;
  loom_type_t base_type = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref(
      body_reader->reader, type_id, IREE_SV("value_type"), type_offset,
      &base_type));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();

  loom_type_t type = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_bind_type(
      body_reader, cursor, base_type, dim_binding_count, &type));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();

  loom_string_id_t value_name_id =
      name_id == 0 ? LOOM_STRING_ID_INVALID : (loom_string_id_t)name_id;
  loom_value_id_t value_id =
      body_reader->value_map[body_reader->next_value_number];
  if (body_reader->next_value_number < body_reader->predefined_value_count) {
    if (value_id >= body_reader->reader->output_module->values.count) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, name_offset,
          IREE_SV("predefined_value_is_not_present_in_the_module"));
    }
    const loom_value_t* value =
        &body_reader->reader->output_module->values.entries[value_id];
    if (!loom_type_equal(value->type, type)) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, name_offset,
          IREE_SV("predefined_value_type_does_not_match_body_value_type"));
    }
    if (value->name_id != value_name_id) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, name_offset,
          IREE_SV("predefined_value_name_does_not_match_body_value_name"));
    }
    body_reader->value_map[body_reader->next_value_number++] = value_id;
    ++body_reader->counts.value_count;
    *out_value_id = value_id;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_module_set_value_type(
      body_reader->reader->output_module, value_id, type));
  if (value_name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_set_value_name(
        body_reader->reader->output_module, value_id, value_name_id));
  }
  ++body_reader->next_value_number;
  ++body_reader->counts.value_count;
  *out_value_id = value_id;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_reader_reserve_value_defs(
    loom_bytecode_body_reader_t* body_reader, uint64_t count,
    loom_value_id_t* out_value_ids) {
  if (body_reader->next_value_number > body_reader->value_capacity ||
      count > body_reader->value_capacity - body_reader->next_value_number) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, body_reader->body_offset,
        IREE_SV("function_body_defines_more_values_than_its_summary"));
  }
  uint64_t start_value_number = body_reader->next_value_number;
  uint64_t end_value_number = start_value_number + count;
  for (uint64_t value_number = start_value_number;
       value_number < end_value_number; ++value_number) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    if (value_number < body_reader->predefined_value_count) {
      value_id = body_reader->predefined_values[value_number];
      if (value_id >= body_reader->reader->output_module->values.count) {
        return loom_bytecode_reader_emit_invalid_ir_body(
            body_reader, body_reader->body_offset,
            IREE_SV("predefined_value_is_not_present_in_the_module"));
      }
    } else {
      IREE_RETURN_IF_ERROR(loom_module_define_value(
          body_reader->reader->output_module, loom_type_none(), &value_id));
    }
    body_reader->value_map[value_number] = value_id;
    if (out_value_ids) {
      out_value_ids[value_number - start_value_number] = value_id;
    }
  }
  body_reader->available_value_count = end_value_number;
  return iree_ok_status();
}

static uint8_t loom_bytecode_instance_flags_mask(
    const loom_op_vtable_t* vtable) {
  if (!iree_all_bits_set(vtable->vtable_flags,
                         LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS)) {
    return 0;
  }
  if (vtable->instance_flags_case_count >= 8) return UINT8_MAX;
  return (uint8_t)((1u << vtable->instance_flags_case_count) - 1u);
}

static iree_status_t loom_bytecode_body_reader_read_op(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    uint32_t depth) {
  uint64_t op_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
      body_reader->reader, op_table_index_plus1, op_offset, &vtable));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();
  loom_op_kind_t op_kind =
      body_reader->reader->op_kinds[op_table_index_plus1 - 1];

  uint8_t instance_flags = 0;
  uint64_t instance_flags_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(body_reader->reader, cursor,
                                                    &instance_flags));
  uint8_t instance_flags_mask = loom_bytecode_instance_flags_mask(vtable);
  if (iree_any_bit_set(instance_flags, (uint8_t)~instance_flags_mask)) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, instance_flags_offset,
        IREE_SV("operation_instance_flags_contain_undeclared_bits"));
  }
  uint64_t location_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(body_reader->reader,
                                                         cursor, &location_id));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();
  if (location_id != 0) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_location_ref(
        body_reader->reader, location_id,
        loom_bytecode_reader_cursor_absolute_position(cursor)));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
  }

  const iree_string_view_t* op_comments = NULL;
  iree_host_size_t op_comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
      body_reader->reader, cursor, body_reader->arena, &op_comments,
      &op_comment_count));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();

  uint64_t operand_count = 0;
  uint64_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &operand_count));
  if (operand_count > UINT16_MAX || operand_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, op_offset, IREE_SV("operand_count_exceeds_field_width"));
  }
  loom_value_id_t* operands = NULL;
  if (operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->arena, (iree_host_size_t)operand_count,
        sizeof(loom_value_id_t), (void**)&operands));
  }
  for (uint64_t i = 0; i < operand_count; ++i) {
    uint64_t ref_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->reader, cursor, &value_number));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_lookup_value(
        body_reader, value_number, body_reader->next_value_number, ref_offset,
        IREE_SV("operand value reference must target a previously defined "
                "value"),
        &operands[i]));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
  }
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  uint16_t* operand_segment_counts = NULL;
  if (operand_segment_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->arena, operand_segment_count, sizeof(uint16_t),
        (void**)&operand_segment_counts));
    uint32_t total_segment_count = 0;
    for (uint8_t i = 0; i < operand_segment_count; ++i) {
      uint64_t segment_count_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t segment_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          body_reader->reader, cursor, &segment_count));
      if (segment_count > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_ir_body(
            body_reader, segment_count_offset,
            IREE_SV("operand_segment_count_exceeds_field_width"));
      }
      const loom_operand_descriptor_t* descriptor =
          &vtable->operand_descriptors[i];
      if (!iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC)) {
        const bool optional =
            iree_any_bit_set(descriptor->flags, LOOM_OPERAND_OPTIONAL);
        if ((!optional && segment_count != 1) ||
            (optional && segment_count > 1)) {
          return loom_bytecode_reader_emit_invalid_ir_body(
              body_reader, segment_count_offset,
              IREE_SV("operand_segment_count_violates_field_arity"));
        }
      }
      operand_segment_counts[i] = (uint16_t)segment_count;
      total_segment_count += (uint32_t)segment_count;
    }
    if (total_segment_count != operand_count) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, op_offset,
          IREE_SV("operand_segment_counts_do_not_sum_to_operand_count"));
    }
  }

  uint64_t successor_count = 0;
  uint64_t successor_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &successor_count));
  if (successor_count > UINT8_MAX || successor_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, successor_count_offset,
        IREE_SV("successor_count_exceeds_field_width"));
  }
  loom_block_t** successors = NULL;
  if (successor_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->arena, (iree_host_size_t)successor_count,
        sizeof(loom_block_t*), (void**)&successors));
    loom_region_t* successor_region =
        builder->ip.block ? builder->ip.block->parent_region : NULL;
    if (!successor_region) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, successor_count_offset,
          IREE_SV("operation_successors_require_an_enclosing_region"));
    }
    for (uint64_t i = 0; i < successor_count; ++i) {
      uint64_t ref_offset =
          loom_bytecode_reader_cursor_absolute_position(cursor);
      uint64_t block_index = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          body_reader->reader, cursor, &block_index));
      if (loom_bytecode_reader_has_errors(body_reader->reader))
        return iree_ok_status();
      if (block_index >= successor_region->block_count) {
        return loom_bytecode_reader_emit_invalid_ir_body(
            body_reader, ref_offset,
            IREE_SV("successor_block_index_is_out_of_range"));
      }
      successors[i] =
          loom_region_block(successor_region, (uint16_t)block_index);
    }
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &result_count));
  if (result_count > UINT16_MAX || result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, op_offset, IREE_SV("result_count_exceeds_field_width"));
  }
  loom_value_id_t* results = NULL;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->arena, (iree_host_size_t)result_count,
        sizeof(loom_value_id_t), (void**)&results));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_reserve_value_defs(
      body_reader, result_count, results));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();
  for (uint64_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_define_value(
        body_reader, cursor, &results[i]));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
  }

  uint64_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &tied_result_count));
  if (tied_result_count > UINT16_MAX ||
      tied_result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, op_offset,
        IREE_SV("tied_result_count_exceeds_field_width"));
  }
  loom_tied_result_t* tied_results = NULL;
  if (tied_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        body_reader->arena, (iree_host_size_t)tied_result_count,
        sizeof(loom_tied_result_t), (void**)&tied_results));
  }
  for (uint64_t i = 0; i < tied_result_count; ++i) {
    uint64_t tie_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t result_index = 0;
    uint64_t operand_index = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->reader, cursor, &result_index));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        body_reader->reader, cursor, &operand_index));
    if (result_index >= result_count || operand_index >= operand_count) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, tie_offset,
          IREE_SV("tied_result_references_an_out_of_range_operand_or_result"));
    }
    tied_results[i] = (loom_tied_result_t){
        .result_index = (uint16_t)result_index,
        .operand_index = (uint16_t)operand_index,
    };
  }

  uint64_t present_attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &present_attr_count));
  if (present_attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, op_offset,
        IREE_SV("present_attribute_count_exceeds_op_attribute_slots"));
  }
  loom_attribute_t* attrs = NULL;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(body_reader->arena, vtable->attribute_count,
                                  sizeof(loom_attribute_t), (void**)&attrs));
    memset(attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  for (uint64_t i = 0; i < present_attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(body_reader->reader,
                                                           cursor, &key_id));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        body_reader->reader, key_id, IREE_SV("attribute_key"), key_offset,
        &key));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    const loom_attr_descriptor_t* descriptor = NULL;
    uint8_t attr_index = 0;
    for (; attr_index < vtable->attribute_count; ++attr_index) {
      const loom_attr_descriptor_t* candidate =
          &vtable->attr_descriptors[attr_index];
      if (iree_string_view_equal(key, loom_attr_descriptor_name(candidate))) {
        descriptor = candidate;
        break;
      }
    }
    if (!descriptor) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, key_offset,
          IREE_SV("attribute_key_is_not_declared_by_the_op"));
    }
    if (!loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_reader_emit_invalid_ir_body(
          body_reader, key_offset,
          IREE_SV("attribute_key_appears_more_than_once"));
    }
    uint8_t value_kind = 0;
    uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(body_reader->reader, cursor, &value_kind));
    if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
      return loom_bytecode_reader_emit_enum_value(
          body_reader->reader, IREE_SV("attribute_kind"), value_kind,
          LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_attr_value(
        body_reader->reader, cursor, body_reader, descriptor, value_kind,
        &attrs[attr_index]));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
  }

  uint64_t region_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      body_reader->reader, cursor, &region_count));
  if (region_count > UINT8_MAX || region_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, op_offset, IREE_SV("region_count_exceeds_field_width"));
  }

  loom_op_t* op = NULL;
  if (operand_segment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op_with_successors(
        builder, op_kind, (uint16_t)operand_count, operand_segment_counts,
        operand_segment_count, (uint16_t)result_count, (uint8_t)successor_count,
        (uint8_t)region_count, (uint16_t)tied_result_count,
        vtable->attribute_count, (loom_location_id_t)location_id, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
        builder, op_kind, (uint16_t)operand_count, (uint16_t)result_count,
        (uint8_t)successor_count, (uint8_t)region_count,
        (uint16_t)tied_result_count, vtable->attribute_count,
        (loom_location_id_t)location_id, &op));
  }
  if (operand_count > 0) {
    memcpy(loom_op_operands(op), operands,
           (iree_host_size_t)operand_count * sizeof(loom_value_id_t));
  }
  if (successor_count > 0) {
    memcpy(loom_op_successors(op), successors,
           (iree_host_size_t)successor_count * sizeof(loom_block_t*));
  }
  if (result_count > 0) {
    memcpy(loom_op_results(op), results,
           (iree_host_size_t)result_count * sizeof(loom_value_id_t));
  }
  if (tied_result_count > 0) {
    memcpy(loom_op_tied_results(op), tied_results,
           (iree_host_size_t)tied_result_count * sizeof(loom_tied_result_t));
  }
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }
  op->instance_flags = instance_flags;
  for (uint64_t i = 0; i < region_count; ++i) {
    loom_region_t* region = NULL;
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_read_region(
        body_reader, cursor, builder, op, depth + 1, &region));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    loom_op_regions(op)[i] = region;
  }
  ++body_reader->counts.op_count;
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
  if (op_comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        body_reader->reader->output_module, op, op_comments, op_comment_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_reader_read_block(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    loom_block_t* block, uint32_t depth) {
  uint8_t has_label = 0;
  uint64_t label_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(body_reader->reader, cursor, &has_label));
  if (has_label > 1) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, label_offset,
        IREE_SV("block_has_label_byte_must_be_0_or_1"));
  }
  if (has_label) {
    uint64_t label_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(body_reader->reader,
                                                           cursor, &label_id));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    iree_string_view_t unused_label = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        body_reader->reader, label_id, IREE_SV("block_label"), label_offset,
        &unused_label));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    block->label_id = (loom_string_id_t)label_id;
  }

  const iree_string_view_t* block_comments = NULL;
  iree_host_size_t block_comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
      body_reader->reader, cursor, body_reader->arena, &block_comments,
      &block_comment_count));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();
  if (block_comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_block_comments(
        body_reader->reader->output_module, block, block_comments,
        block_comment_count));
  }

  uint64_t arg_count = 0;
  uint64_t arg_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(body_reader->reader,
                                                         cursor, &arg_count));
  if (arg_count > UINT16_MAX || arg_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, arg_count_offset,
        IREE_SV("block_argument_count_exceeds_field_width"));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_reserve_value_defs(
      body_reader, arg_count, NULL));
  if (loom_bytecode_reader_has_errors(body_reader->reader))
    return iree_ok_status();
  for (uint64_t i = 0; i < arg_count; ++i) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_body_reader_define_value(body_reader, cursor, &value_id));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_block_add_arg(body_reader->reader->output_module,
                                            block, value_id));
  }
  if (body_reader->next_value_number < body_reader->predefined_value_count) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, arg_count_offset,
        IREE_SV("function_body_entry_block_is_missing_signature_arguments"));
  }

  loom_builder_set_block(builder, block);
  uint64_t op_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(body_reader->reader,
                                                         cursor, &op_count));
  if (op_count > UINT16_MAX || op_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, arg_count_offset,
        IREE_SV("block_operation_count_exceeds_field_width"));
  }
  for (uint64_t i = 0; i < op_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_body_reader_read_op(body_reader, cursor, builder, depth));
    if (loom_bytecode_reader_has_errors(body_reader->reader))
      return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_body_reader_read_region(
    loom_bytecode_body_reader_t* body_reader,
    loom_bytecode_reader_cursor_t* cursor, loom_builder_t* builder,
    loom_op_t* parent_op, uint32_t depth, loom_region_t** out_region) {
  if (depth >= LOOM_BYTECODE_MAX_REGION_DEPTH) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("region_nesting_exceeds_bytecode_maximum_depth"));
  }
  uint64_t block_count = 0;
  uint64_t block_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(body_reader->reader,
                                                         cursor, &block_count));
  if (block_count > UINT16_MAX || block_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_ir_body(
        body_reader, block_count_offset,
        IREE_SV("region_block_count_exceeds_field_width"));
  }
  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(
      body_reader->reader->output_module, (uint16_t)block_count, &region));
  ++body_reader->counts.region_count;
  body_reader->counts.block_count += block_count;

  loom_builder_ip_t saved =
      loom_builder_enter_region(builder, parent_op, region);
  for (uint64_t i = 0; i < block_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_read_block(
        body_reader, cursor, builder, loom_region_block(region, (uint16_t)i),
        depth));
    if (loom_bytecode_reader_has_errors(body_reader->reader)) {
      loom_builder_restore(builder, saved);
      return iree_ok_status();
    }
  }
  loom_builder_restore(builder, saved);
  *out_region = region;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_symbol_regions(
    loom_bytecode_reader_state_t* reader, iree_string_view_t symbol_name,
    const loom_bytecode_reader_section_t* ir_section, uint64_t ir_offset,
    uint32_t ir_length, loom_builder_t* builder, loom_op_t* parent_op,
    uint8_t predefined_region_index, const loom_value_id_t* predefined_values,
    uint16_t predefined_value_count) {
  iree_arena_allocator_t body_arena;
  iree_arena_initialize(reader->block_pool, &body_arena);

  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      ir_section->bytes.data + ir_offset, ir_length,
      ir_section->absolute_offset + ir_offset, IREE_SV("IR"), &cursor);
  uint64_t value_count = 0;
  uint64_t expected_region_count = 0;
  uint64_t expected_block_count = 0;
  uint64_t expected_op_count = 0;
  uint64_t root_region_count = 0;
  iree_status_t status =
      loom_bytecode_reader_read_uvarint(reader, &cursor, &value_count);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_reader_read_uvarint(reader, &cursor,
                                               &expected_region_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_reader_read_uvarint(reader, &cursor,
                                               &expected_block_count);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_reader_read_uvarint(reader, &cursor, &expected_op_count);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_reader_read_uvarint(reader, &cursor, &root_region_count);
  }
  loom_bytecode_body_reader_t body_reader = {
      .reader = reader,
      .arena = &body_arena,
      .symbol_name = symbol_name,
      .body_offset = ir_section->absolute_offset + ir_offset,
      .value_capacity = value_count,
      .predefined_values = predefined_values,
      .predefined_value_count = predefined_value_count,
  };
  iree_host_size_t body_length = cursor.cursor.length;
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
      value_count > IREE_HOST_SIZE_MAX) {
    status = loom_bytecode_reader_emit_invalid_ir_body(
        &body_reader, body_reader.body_offset,
        IREE_SV("function_body_value_count_exceeds_host_size"));
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
      predefined_value_count > value_count) {
    status = loom_bytecode_reader_emit_invalid_ir_body(
        &body_reader, body_reader.body_offset,
        IREE_SV("predefined_signature_values_exceed_function_body_values"));
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
      root_region_count == 0) {
    status = loom_bytecode_reader_emit_invalid_ir_body(
        &body_reader, body_reader.body_offset,
        IREE_SV("symbol_region_root_count_must_be_nonzero"));
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
      root_region_count > parent_op->region_count) {
    status = loom_bytecode_reader_emit_invalid_ir_body(
        &body_reader, body_reader.body_offset,
        IREE_SV("symbol_region_root_count_exceeds_op_region_slots"));
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
      (root_region_count > UINT8_MAX ||
       root_region_count > IREE_HOST_SIZE_MAX)) {
    status = loom_bytecode_reader_emit_invalid_ir_body(
        &body_reader, body_reader.body_offset,
        IREE_SV("symbol_region_root_count_exceeds_field_width"));
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
      (value_count > body_length || expected_region_count > body_length ||
       expected_block_count > body_length || expected_op_count > body_length ||
       root_region_count > body_length)) {
    status = loom_bytecode_reader_emit_invalid_ir_body(
        &body_reader, body_reader.body_offset,
        IREE_SV("function_body_allocation_summary_exceeds_body_length"));
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
      value_count > 0) {
    status = iree_arena_allocate_array(
        &body_arena, (iree_host_size_t)value_count, sizeof(loom_value_id_t),
        (void**)&body_reader.value_map);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader)) {
    bool seen_regions[UINT8_MAX + 1] = {0};
    for (uint64_t root_ordinal = 0; root_ordinal < root_region_count;
         ++root_ordinal) {
      uint64_t region_index_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t region_index = 0;
      status =
          loom_bytecode_reader_read_uvarint(reader, &cursor, &region_index);
      if (!iree_status_is_ok(status) ||
          loom_bytecode_reader_has_errors(reader)) {
        break;
      }
      if (region_index >= parent_op->region_count || region_index > UINT8_MAX) {
        status = loom_bytecode_reader_emit_invalid_ir_body(
            &body_reader, region_index_offset,
            IREE_SV("root_region_index_is_out_of_range_for_symbol_op"));
        break;
      }
      if (seen_regions[region_index]) {
        status = loom_bytecode_reader_emit_invalid_ir_body(
            &body_reader, region_index_offset,
            IREE_SV("root_region_index_appears_more_than_once"));
        break;
      }
      if (predefined_region_index != LOOM_REGION_INDEX_NONE &&
          root_ordinal == 0 && region_index != predefined_region_index) {
        status = loom_bytecode_reader_emit_invalid_ir_body(
            &body_reader, region_index_offset,
            IREE_SV("funclike_body_region_must_be_the_first_root_region"));
        break;
      }
      seen_regions[region_index] = true;

      body_reader.predefined_values =
          region_index == predefined_region_index ? predefined_values : NULL;
      body_reader.predefined_value_count =
          region_index == predefined_region_index ? predefined_value_count : 0;
      loom_region_t* region = NULL;
      status = loom_bytecode_body_reader_read_region(
          &body_reader, &cursor, builder, parent_op, 0, &region);
      if (!iree_status_is_ok(status) ||
          loom_bytecode_reader_has_errors(reader)) {
        break;
      }
      loom_op_regions(parent_op)[region_index] = region;
    }
    if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader) &&
        predefined_region_index != LOOM_REGION_INDEX_NONE &&
        predefined_region_index < parent_op->region_count &&
        !seen_regions[predefined_region_index]) {
      status = loom_bytecode_reader_emit_invalid_ir_body(
          &body_reader, body_reader.body_offset,
          IREE_SV("symbol_region_payload_is_missing_the_funclike_body"));
    }
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader)) {
    if (body_reader.counts.value_count != value_count ||
        body_reader.counts.region_count != expected_region_count ||
        body_reader.counts.block_count != expected_block_count ||
        body_reader.counts.op_count != expected_op_count) {
      status = loom_bytecode_reader_emit_invalid_ir_body(
          &body_reader, body_reader.body_offset,
          IREE_SV("function_body_allocation_summary_does_not_match_ir"));
    }
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(reader)) {
    status = loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("IR"));
  }
  iree_arena_deinitialize(&body_arena);
  return status;
}

static loom_symbol_kind_t loom_bytecode_reader_decode_symbol_kind(
    uint8_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_SYMBOL_FUNC_DEF:
      return LOOM_SYMBOL_FUNC_DEF;
    case LOOM_BYTECODE_SYMBOL_FUNC_DECL:
      return LOOM_SYMBOL_FUNC_DECL;
    case LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE:
      return LOOM_SYMBOL_FUNC_TEMPLATE;
    case LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL:
      return LOOM_SYMBOL_FUNC_UKERNEL;
    case LOOM_BYTECODE_SYMBOL_GLOBAL:
      return LOOM_SYMBOL_GLOBAL;
    case LOOM_BYTECODE_SYMBOL_EXECUTABLE:
      return LOOM_SYMBOL_EXECUTABLE;
    case LOOM_BYTECODE_SYMBOL_RECORD:
      return LOOM_SYMBOL_RECORD;
    default:
      return LOOM_SYMBOL_NONE;
  }
}

static iree_status_t loom_bytecode_reader_validate_func_enum(
    loom_bytecode_reader_state_t* reader, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint8_t attr_index,
    iree_string_view_t field_name, uint8_t value, uint64_t offset) {
  if (value == 0) return iree_ok_status();
  if (!vtable || attr_index == LOOM_ATTR_INDEX_NONE ||
      attr_index >= vtable->attribute_count || !vtable->attr_descriptors) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index, field_name,
        offset, IREE_SV("function_metadata_field_is_not_supported_by_op"));
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[attr_index];
  if (descriptor->attr_kind != LOOM_ATTR_ENUM) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index, field_name,
        offset, IREE_SV("function_metadata_field_must_target_an_enum_attr"));
  }
  if (descriptor->enum_case_count > 0 && value >= descriptor->enum_case_count) {
    return loom_bytecode_reader_emit_enum_value(
        reader, field_name, value, descriptor->enum_case_count, offset);
  }
  return iree_ok_status();
}

static uint8_t loom_bytecode_find_op_attr_index_by_name(
    const loom_op_vtable_t* vtable, iree_string_view_t name) {
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (iree_string_view_equal(
            loom_attr_descriptor_name(&vtable->attr_descriptors[i]), name)) {
      return i;
    }
  }
  return LOOM_ATTR_INDEX_NONE;
}

static bool loom_bytecode_attr_is_symbol_identity(
    const loom_op_vtable_t* vtable, uint8_t attr_index) {
  return vtable && vtable->symbol_def &&
         attr_index == vtable->symbol_def->name_attr_index;
}

static bool loom_bytecode_func_metadata_attr_is_shared(
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    uint8_t attr_index) {
  if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) return true;
  iree_string_view_t name =
      loom_attr_descriptor_name(&vtable->attr_descriptors[attr_index]);
  if (iree_string_view_equal(name, IREE_SV("import_module")) ||
      iree_string_view_equal(name, IREE_SV("import_symbol")) ||
      attr_index == func_like->visibility_attr_index ||
      attr_index == func_like->cc_attr_index ||
      attr_index == func_like->purity_attr_index ||
      attr_index == func_like->predicates_attr_index) {
    return true;
  }
  if ((vtable->symbol_kind == LOOM_SYMBOL_FUNC_TEMPLATE ||
       vtable->symbol_kind == LOOM_SYMBOL_FUNC_UKERNEL) &&
      (attr_index == func_like->implements_attr_index ||
       attr_index == func_like->priority_attr_index)) {
    return true;
  }
  return false;
}

static iree_status_t loom_bytecode_reader_read_func_payload_attrs(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    loom_bytecode_body_reader_t* signature_reader, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    loom_attribute_t* attrs) {
  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_function_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, key_id, IREE_SV("function_attribute_key"), key_offset, &key));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    uint8_t attr_index = loom_bytecode_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_func_metadata_attr_is_shared(vtable, func_like,
                                                   attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_is_reconstructed_from_shared_metadata"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit) ||
        !loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    const loom_attr_descriptor_t* descriptor =
        &vtable->attr_descriptors[attr_index];
    uint8_t value_kind = 0;
    uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
    if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), value_kind,
          LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_attr_value(
        reader, cursor, signature_reader, descriptor, value_kind,
        &attrs[attr_index]));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_func_payload_attrs(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t symbol_index, const loom_op_vtable_t* vtable,
    const loom_func_like_vtable_t* func_like) {
  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_function_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, key_id, IREE_SV("function_attribute_key"), key_offset, &key));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    uint8_t attr_index = loom_bytecode_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_func_metadata_attr_is_shared(vtable, func_like,
                                                   attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_is_reconstructed_from_shared_metadata"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("function_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    uint8_t value_kind = 0;
    uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
    if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), value_kind,
          LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_attr_value(
        reader, cursor, value_kind, false));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return iree_ok_status();
}

static uint8_t loom_bytecode_find_symbol_attr_index(
    const loom_op_vtable_t* vtable) {
  if (!vtable || !vtable->attr_descriptors) return LOOM_ATTR_INDEX_NONE;
  if (vtable->symbol_def) {
    uint8_t attr_index = vtable->symbol_def->name_attr_index;
    if (attr_index < vtable->attribute_count &&
        vtable->attr_descriptors[attr_index].attr_kind == LOOM_ATTR_SYMBOL) {
      return attr_index;
    }
    return LOOM_ATTR_INDEX_NONE;
  }
  if (!vtable->attr_descriptors) return LOOM_ATTR_INDEX_NONE;
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (vtable->attr_descriptors[i].attr_kind == LOOM_ATTR_SYMBOL) return i;
  }
  return LOOM_ATTR_INDEX_NONE;
}

static iree_status_t loom_bytecode_reader_validate_global_vtable(
    loom_bytecode_reader_state_t* reader, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint64_t op_ref_offset) {
  if (!iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      !vtable->symbol_def ||
      vtable->symbol_def->bytecode_kind != LOOM_SYMBOL_GLOBAL) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("global-payload symbol defining op must use the GLOBAL "
                "bytecode payload"));
  }
  if (loom_op_vtable_operand_descriptor_count(vtable) != 0 ||
      vtable->region_count != 0 ||
      iree_any_bit_set(
          vtable->vtable_flags,
          LOOM_OP_VTABLE_VARIADIC_OPERANDS | LOOM_OP_VTABLE_VARIADIC_REGIONS)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("global_symbol_defining_op_must_not_have_operands_or_regions"));
  }
  if (loom_bytecode_find_symbol_attr_index(vtable) == LOOM_ATTR_INDEX_NONE) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("global_symbol_defining_op_must_declare_a_symbol_attribute"));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_validate_record_vtable(
    loom_bytecode_reader_state_t* reader, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint64_t op_ref_offset) {
  if (!iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      !vtable->symbol_def ||
      !loom_symbol_definition_implements(vtable->symbol_def,
                                         LOOM_SYMBOL_INTERFACE_RECORD)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record_symbol_defining_op_must_define_a_record_symbol"));
  }
  if (loom_op_vtable_operand_descriptor_count(vtable) != 0 ||
      vtable->fixed_result_count != 0 ||
      iree_any_bit_set(
          vtable->vtable_flags,
          LOOM_OP_VTABLE_VARIADIC_OPERANDS | LOOM_OP_VTABLE_VARIADIC_RESULTS)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record_symbol_defining_op_must_not_have_operands_or_results"));
  }
  if (vtable->region_count > 1 ||
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record symbol defining op must declare at most one fixed "
                "region"));
  }
  if (loom_bytecode_find_symbol_attr_index(vtable) == LOOM_ATTR_INDEX_NONE) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record_symbol_defining_op_must_declare_a_symbol_attribute"));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_value_def(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    iree_string_view_t field_name) {
  uint64_t name_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t name_id = 0;
  uint64_t type_id = 0;
  uint64_t dim_binding_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &name_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &type_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &dim_binding_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (name_id != 0) {
    iree_string_view_t unused_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("value_name"), name_offset, &unused_name));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  loom_type_t unused_type = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_type_ref(
      reader, type_id, field_name, name_offset, &unused_type));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  for (uint64_t i = 0; i < dim_binding_count; ++i) {
    int64_t unused_value_ref = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_svarint(reader, cursor, &unused_value_ref));
  }
  uint64_t unused_encoding_ref = 0;
  return loom_bytecode_reader_read_uvarint(reader, cursor,
                                           &unused_encoding_ref);
}

static iree_status_t loom_bytecode_reader_skip_global_payload(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t symbol_index, loom_bytecode_symbol_metadata_t* symbol_metadata) {
  uint64_t op_ref_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
      reader, op_table_index_plus1, op_ref_offset, &vtable));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_global_vtable(
      reader, symbol_index, vtable, op_ref_offset));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (symbol_metadata) {
    symbol_metadata->defining_op_name = loom_op_vtable_name(vtable);
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
      reader, cursor, reader->arena, NULL, NULL));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  uint64_t result_count = 0;
  uint64_t result_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &result_count));
  if (result_count == 0 || result_count > UINT16_MAX ||
      result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("result_count"), result_count_offset,
        IREE_SV("global_result_count_must_be_non_zero_and_fit_in_uint16_t"));
  }
  if (symbol_metadata) {
    symbol_metadata->result_count = (uint16_t)result_count;
  }
  uint64_t local_value_count = 0;
  uint64_t local_value_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &local_value_count));
  if (local_value_count < result_count ||
      local_value_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("local_value_count"), local_value_count_offset,
        IREE_SV("global_local_value_count_must_cover_all_op_results"));
  }
  if (symbol_metadata) {
    symbol_metadata->local_value_count = local_value_count;
  }
  for (uint64_t i = 0; i < local_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_value_def(
        reader, cursor, IREE_SV("global_local_value_type")));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_global_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, key_id, IREE_SV("global_attribute_key"), key_offset, &key));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    uint8_t attr_index = loom_bytecode_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    uint8_t value_kind = 0;
    uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
    if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), value_kind,
          LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_attr_value(
        reader, cursor, value_kind, false));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_record_payload(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_reader_section_t* ir_section, uint64_t symbol_index,
    loom_bytecode_symbol_metadata_t* symbol_metadata) {
  uint64_t op_ref_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
      reader, op_table_index_plus1, op_ref_offset, &vtable));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_record_vtable(
      reader, symbol_index, vtable, op_ref_offset));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (symbol_metadata) {
    symbol_metadata->defining_op_name = loom_op_vtable_name(vtable);
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
      reader, cursor, reader->arena, NULL, NULL));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_record_attribute_count_exceeds_op_attribute_slots"));
  }
  uint64_t seen_attr_bits[4] = {0};
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, key_id, IREE_SV("record_attribute_key"), key_offset, &key));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    uint8_t attr_index = loom_bytecode_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_is_not_declared_by_the_op"));
    }
    if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    uint64_t attr_bit = (uint64_t)1 << (attr_index % 64);
    uint64_t* attr_word = &seen_attr_bits[attr_index / 64];
    if (iree_any_bit_set(*attr_word, attr_bit)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_appears_more_than_once"));
    }
    *attr_word |= attr_bit;
    uint8_t value_kind = 0;
    uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
    if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), value_kind,
          LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_attr_value(
        reader, cursor, value_kind, false));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  uint8_t has_body = 0;
  uint64_t has_body_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &has_body));
  if (has_body > 1) {
    return loom_bytecode_reader_emit_enum_value(reader, IREE_SV("has_body"),
                                                has_body, 2, has_body_offset);
  }
  if ((vtable->region_count == 1) != (has_body != 0)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("has_body"), has_body_offset,
        IREE_SV("record_body_presence_must_match_the_defining_op_region"));
  }
  if (has_body) {
    uint64_t ir_offset = 0;
    uint32_t ir_length = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &ir_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(reader, cursor, &ir_length));
    if (ir_section) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
          reader, IREE_SV("IR body"), ir_offset, ir_length,
          ir_section->length));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    }
    if (symbol_metadata) {
      symbol_metadata->has_body = true;
      symbol_metadata->body_offset = ir_offset;
      symbol_metadata->body_absolute_offset =
          ir_section ? ir_section->absolute_offset + ir_offset : 0;
      symbol_metadata->body_length = ir_length;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_skip_symbol_payload(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint8_t kind, uint64_t symbol_index) {
  if (kind <= LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
    uint64_t op_ref_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t op_table_index_plus1 = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        reader, cursor, &op_table_index_plus1));
    const loom_op_vtable_t* vtable = NULL;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
        reader, op_table_index_plus1, op_ref_offset, &vtable));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (!vtable->func_like) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("def_op_table_index_plus1"), op_ref_offset,
          IREE_SV("function_symbol_defining_op_must_implement_funclike"));
    }

    uint64_t value = 0;
    uint8_t byte_value = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
        reader, cursor, reader->arena, NULL, NULL));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &byte_value));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &byte_value));
    uint64_t arg_count = 0;
    uint64_t result_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &arg_count));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &result_count));
    for (uint64_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_value_def(
          reader, cursor, IREE_SV("arg_type")));
    }
    for (uint64_t i = 0; i < result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, cursor, &byte_value));
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_value_def(
          reader, cursor, IREE_SV("result_type")));
      if (byte_value) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, cursor, &value));
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &value));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_skip_predicate_list(reader, cursor, false));
    if (kind == LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE ||
        kind == LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &value));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &value));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_func_payload_attrs(
        reader, cursor, symbol_index, vtable, vtable->func_like));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &byte_value));
    if (byte_value) {
      uint64_t offset = 0;
      uint32_t length = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u64_le(reader, cursor, &offset));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u32_le(reader, cursor, &length));
    }
  } else if (kind == LOOM_BYTECODE_SYMBOL_GLOBAL) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_global_payload(
        reader, cursor, symbol_index, /*symbol_metadata=*/NULL));
  } else if (kind == LOOM_BYTECODE_SYMBOL_RECORD) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_record_payload(
        reader, cursor, /*ir_section=*/NULL, symbol_index,
        /*symbol_metadata=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_symbol_cursor_to_entries(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* symbols_section,
    loom_bytecode_reader_cursor_t* cursor) {
  loom_bytecode_reader_cursor_initialize(
      symbols_section->bytes.data, symbols_section->bytes.data_length,
      symbols_section->absolute_offset, IREE_SV("SYMBOLS"), cursor);
  uint64_t count = 0;
  uint64_t import_count = 0;
  uint64_t export_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &import_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &export_count));
  for (uint64_t i = 0; i < import_count + export_count; ++i) {
    uint64_t unused_offset = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &unused_offset));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_predeclare_symbols(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* symbols_section) {
  loom_bytecode_reader_cursor_t cursor;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_symbol_cursor_to_entries(
      reader, symbols_section, &cursor));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  for (iree_host_size_t i = 0; i < reader->symbol_count; ++i) {
    uint64_t name_id = 0;
    uint8_t kind = 0;
    uint8_t visibility = 0;
    uint16_t flags = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, &cursor, &visibility));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &flags));
    (void)visibility;
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT) {
      uint64_t unused = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &unused));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &unused));
    }
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(
        reader->output_module, (loom_string_id_t)name_id, &symbol_id));
    loom_symbol_t* symbol = &reader->output_module->symbols.entries[symbol_id];
    symbol->kind = loom_bytecode_reader_decode_symbol_kind(kind);
    symbol->flags = 0;
    if (flags & LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC) {
      symbol->flags |= LOOM_SYMBOL_FLAG_PUBLIC;
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_skip_symbol_payload(reader, &cursor, kind, i));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_function_symbol(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_reader_section_t* ir_section, uint64_t name_id,
    uint8_t kind, uint16_t flags, loom_string_id_t import_module_id,
    loom_string_id_t import_symbol_id, loom_builder_t* builder) {
  uint16_t symbol_id =
      loom_module_find_symbol(reader->output_module, (loom_string_id_t)name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), 0, IREE_SV("name_id"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("function_symbol_was_not_predeclared"));
  }
  loom_symbol_ref_t callee_ref = {0, symbol_id};
  iree_string_view_t symbol_name = reader->strings[name_id];

  uint64_t op_ref_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
      reader, op_table_index_plus1, op_ref_offset, &vtable));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (!vtable->func_like) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("function_symbol_defining_op_must_implement_funclike"));
  }
  loom_op_kind_t op_kind = reader->op_kinds[op_table_index_plus1 - 1];
  const loom_func_like_vtable_t* func_like = vtable->func_like;

  const iree_string_view_t* symbol_comments = NULL;
  iree_host_size_t symbol_comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
      reader, cursor, reader->arena, &symbol_comments, &symbol_comment_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  uint8_t calling_convention = 0;
  uint64_t calling_convention_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(reader, cursor, &calling_convention));
  uint8_t purity = 0;
  uint64_t purity_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &purity));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_func_enum(
      reader, symbol_id, vtable, func_like->cc_attr_index,
      IREE_SV("calling_convention"), calling_convention,
      calling_convention_offset));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_func_enum(
      reader, symbol_id, vtable, func_like->purity_attr_index,
      IREE_SV("purity"), purity, purity_offset));
  uint64_t arg_count = 0;
  uint64_t result_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &arg_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &result_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (arg_count > UINT16_MAX || result_count > UINT16_MAX ||
      arg_count > IREE_HOST_SIZE_MAX || result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
        IREE_SV("signature_count"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("function_signature_count_exceeds_field_width"));
  }

  uint64_t signature_value_count = arg_count + result_count;
  loom_value_id_t* signature_values = NULL;
  if (signature_value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)signature_value_count,
        sizeof(loom_value_id_t), (void**)&signature_values));
  }
  loom_bytecode_body_reader_t signature_reader = {
      .reader = reader,
      .arena = reader->arena,
      .symbol_name = symbol_name,
      .body_offset = loom_bytecode_reader_cursor_absolute_position(cursor),
      .value_map = signature_values,
      .value_capacity = signature_value_count,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_reserve_value_defs(
      &signature_reader, signature_value_count, signature_values));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  for (uint64_t i = 0; i < arg_count; ++i) {
    loom_value_id_t arg_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_define_value(
        &signature_reader, cursor, &arg_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)result_count,
        sizeof(loom_tied_result_t), (void**)&tied_results));
  }
  for (uint64_t i = 0; i < result_count; ++i) {
    uint64_t result_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint8_t is_tied = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &is_tied));
    loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_define_value(
        &signature_reader, cursor, &result_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (is_tied) {
      uint64_t operand_index = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, cursor, &operand_index));
      if (operand_index >= arg_count) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
            IREE_SV("tied_operand_index"), result_offset,
            IREE_SV("tied_result_references_an_argument_out_of_range"));
      }
      tied_results[tied_result_count++] = (loom_tied_result_t){
          .result_index = (uint16_t)i,
          .operand_index = (uint16_t)operand_index,
      };
    }
  }
  uint64_t encoded_tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
      reader, cursor, &encoded_tied_result_count));
  if (encoded_tied_result_count != tied_result_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
        IREE_SV("tied_result_count"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("tied_result_summary_does_not_match_tied_result_records"));
  }

  loom_attribute_t predicates_attr = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_predicate_list_attr(
      reader, cursor, &signature_reader, &predicates_attr));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  loom_string_id_t implements_id = LOOM_STRING_ID_INVALID;
  int64_t priority = 0;
  if (kind == LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE ||
      kind == LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
    uint64_t implements_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t implements_string_id = 0;
    uint64_t priority_value = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
        reader, cursor, &implements_string_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &priority_value));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t unused = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, implements_string_id, IREE_SV("implements_op_name"),
        implements_offset, &unused));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    implements_id = (loom_string_id_t)implements_string_id;
    priority = (int64_t)priority_value;
  }

  loom_attribute_t* func_attrs = NULL;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, vtable->attribute_count, sizeof(loom_attribute_t),
        (void**)&func_attrs));
    memset(func_attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_func_payload_attrs(
      reader, cursor, &signature_reader, symbol_id, vtable, func_like,
      func_attrs));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  func_attrs[func_like->callee_attr_index] = loom_attr_symbol(callee_ref);
  if ((flags & LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC) &&
      func_like->visibility_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->visibility_attr_index] = loom_attr_enum(1);
  }
  if (calling_convention != 0 &&
      func_like->cc_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->cc_attr_index] = loom_attr_enum(calling_convention);
  }
  if (purity != 0 && func_like->purity_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->purity_attr_index] = loom_attr_enum(purity);
  }
  if (import_module_id != LOOM_STRING_ID_INVALID) {
    uint8_t import_module_attr_index = loom_bytecode_find_op_attr_index_by_name(
        vtable, IREE_SV("import_module"));
    if (import_module_attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
          IREE_SV("import_module"), 0,
          IREE_SV("imported_symbol_op_has_no_import_module_attr"));
    }
    func_attrs[import_module_attr_index] = loom_attr_string(import_module_id);
    if (iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL)) {
      uint8_t import_symbol_attr_index =
          loom_bytecode_find_op_attr_index_by_name(vtable,
                                                   IREE_SV("import_symbol"));
      if (import_symbol_attr_index == LOOM_ATTR_INDEX_NONE) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_id,
            IREE_SV("import_symbol"), 0,
            IREE_SV("imported_symbol_op_has_no_import_symbol_attr"));
      }
      func_attrs[import_symbol_attr_index] = loom_attr_string(import_symbol_id);
    }
  }
  if (!loom_attr_is_absent(predicates_attr) && predicates_attr.count != 0 &&
      func_like->predicates_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->predicates_attr_index] = predicates_attr;
  }
  if (implements_id != LOOM_STRING_ID_INVALID &&
      func_like->implements_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->implements_attr_index] =
        loom_attr_string(implements_id);
  }
  if (priority != 0 && func_like->priority_attr_index != LOOM_ATTR_INDEX_NONE) {
    func_attrs[func_like->priority_attr_index] = loom_attr_i64(priority);
  }

  uint8_t has_body = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &has_body));
  uint64_t ir_offset = 0;
  uint32_t ir_length = 0;
  if (has_body) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &ir_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(reader, cursor, &ir_length));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("IR body"), ir_offset, ir_length,
        ir_section ? ir_section->length : 0));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  uint16_t operand_count =
      func_like->args_as_operands ? (uint16_t)arg_count : 0;
  uint8_t region_count = has_body ? vtable->region_count : 0;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, op_kind, operand_count, (uint16_t)result_count, region_count,
      tied_result_count, vtable->attribute_count, LOOM_LOCATION_NONE, &op));
  if (func_like->args_as_operands) {
    for (uint64_t i = 0; i < arg_count; ++i) {
      loom_op_operands(op)[i] = signature_values[i];
    }
  }
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), func_attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }

  for (uint64_t i = 0; i < result_count; ++i) {
    loom_op_results(op)[i] = signature_values[arg_count + i];
  }
  if (tied_result_count > 0) {
    memcpy(loom_op_tied_results(op), tied_results,
           tied_result_count * sizeof(loom_tied_result_t));
  }
  if (has_body) {
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_symbol_regions(
        reader, symbol_name, ir_section, ir_offset, ir_length, builder, op,
        func_like->body_region_index, signature_values, (uint16_t)arg_count));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
  if (symbol_comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        reader->output_module, op, symbol_comments, symbol_comment_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_global_symbol(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t name_id, uint64_t symbol_index, loom_builder_t* builder) {
  uint16_t symbol_id =
      loom_module_find_symbol(reader->output_module, (loom_string_id_t)name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("name_id"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("global_symbol_was_not_predeclared"));
  }
  loom_symbol_ref_t symbol_ref = {0, symbol_id};
  iree_string_view_t symbol_name = reader->strings[name_id];

  uint64_t op_ref_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
      reader, op_table_index_plus1, op_ref_offset, &vtable));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_global_vtable(
      reader, symbol_index, vtable, op_ref_offset));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  loom_op_kind_t op_kind = reader->op_kinds[op_table_index_plus1 - 1];

  const iree_string_view_t* symbol_comments = NULL;
  iree_host_size_t symbol_comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
      reader, cursor, reader->arena, &symbol_comments, &symbol_comment_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  uint64_t result_count = 0;
  uint64_t result_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &result_count));
  if (result_count == 0 || result_count > UINT16_MAX ||
      result_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("result_count"), result_count_offset,
        IREE_SV("global_result_count_must_be_non_zero_and_fit_in_uint16_t"));
  }
  uint64_t local_value_count = 0;
  uint64_t local_value_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &local_value_count));
  if (local_value_count < result_count ||
      local_value_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("local_value_count"), local_value_count_offset,
        IREE_SV("global_local_value_count_must_cover_all_op_results"));
  }

  loom_value_id_t* local_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      reader->arena, (iree_host_size_t)local_value_count,
      sizeof(loom_value_id_t), (void**)&local_values));
  loom_bytecode_body_reader_t global_reader = {
      .reader = reader,
      .arena = reader->arena,
      .symbol_name = symbol_name,
      .body_offset = loom_bytecode_reader_cursor_absolute_position(cursor),
      .value_map = local_values,
      .value_capacity = local_value_count,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_reserve_value_defs(
      &global_reader, local_value_count, local_values));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  for (uint64_t i = 0; i < local_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_body_reader_define_value(
        &global_reader, cursor, &local_values[i]));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  loom_attribute_t* attrs = NULL;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, vtable->attribute_count,
                                  sizeof(loom_attribute_t), (void**)&attrs));
    memset(attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  uint8_t symbol_attr_index = loom_bytecode_find_symbol_attr_index(vtable);
  attrs[symbol_attr_index] = loom_attr_symbol(symbol_ref);

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_global_attribute_count_exceeds_op_attribute_slots"));
  }
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, key_id, IREE_SV("global_attribute_key"), key_offset, &key));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    uint8_t attr_index = loom_bytecode_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_is_not_declared_by_the_op"));
    }
    const loom_attr_descriptor_t* descriptor =
        &vtable->attr_descriptors[attr_index];
    if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    if (!loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("global_attribute_key_appears_more_than_once"));
    }
    uint8_t value_kind = 0;
    uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
    if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), value_kind,
          LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_attr_value(
        reader, cursor, &global_reader, descriptor, value_kind,
        &attrs[attr_index]));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, op_kind, 0, (uint16_t)result_count, 0, 0,
      vtable->attribute_count, LOOM_LOCATION_NONE, &op));
  memcpy(loom_op_results(op), local_values,
         (iree_host_size_t)result_count * sizeof(loom_value_id_t));
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
  if (symbol_comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        reader->output_module, op, symbol_comments, symbol_comment_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_record_symbol(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_reader_section_t* ir_section, uint64_t name_id,
    uint64_t symbol_index, loom_builder_t* builder) {
  uint16_t symbol_id =
      loom_module_find_symbol(reader->output_module, (loom_string_id_t)name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("name_id"),
        loom_bytecode_reader_cursor_absolute_position(cursor),
        IREE_SV("record_symbol_was_not_predeclared"));
  }
  loom_symbol_ref_t symbol_ref = {0, symbol_id};

  uint64_t op_ref_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t op_table_index_plus1 = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &op_table_index_plus1));
  const loom_op_vtable_t* vtable = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
      reader, op_table_index_plus1, op_ref_offset, &vtable));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_record_vtable(
      reader, symbol_index, vtable, op_ref_offset));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  loom_op_kind_t op_kind = reader->op_kinds[op_table_index_plus1 - 1];

  const iree_string_view_t* symbol_comments = NULL;
  iree_host_size_t symbol_comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
      reader, cursor, reader->arena, &symbol_comments, &symbol_comment_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  loom_attribute_t* attrs = NULL;
  if (vtable->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, vtable->attribute_count,
                                  sizeof(loom_attribute_t), (void**)&attrs));
    memset(attrs, 0, vtable->attribute_count * sizeof(loom_attribute_t));
  }
  uint8_t symbol_attr_index = loom_bytecode_find_symbol_attr_index(vtable);
  attrs[symbol_attr_index] = loom_attr_symbol(symbol_ref);

  uint64_t attr_count = 0;
  uint64_t attr_count_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, cursor, &attr_count));
  if (attr_count > vtable->attribute_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("attr_count"), attr_count_offset,
        IREE_SV("present_record_attribute_count_exceeds_op_attribute_slots"));
  }
  for (uint64_t i = 0; i < attr_count; ++i) {
    uint64_t key_offset = loom_bytecode_reader_cursor_absolute_position(cursor);
    uint64_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, cursor, &key_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, key_id, IREE_SV("record_attribute_key"), key_offset, &key));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    uint8_t attr_index = loom_bytecode_find_op_attr_index_by_name(vtable, key);
    if (attr_index == LOOM_ATTR_INDEX_NONE) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_is_not_declared_by_the_op"));
    }
    const loom_attr_descriptor_t* descriptor =
        &vtable->attr_descriptors[attr_index];
    if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record identity symbol attribute is reconstructed from "
                  "name_id"));
    }
    if (!loom_attr_is_absent(attrs[attr_index])) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("attribute_key"), key_offset,
          IREE_SV("record_attribute_key_appears_more_than_once"));
    }
    uint8_t value_kind = 0;
    uint64_t value_kind_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &value_kind));
    if (value_kind >= LOOM_BYTECODE_ATTR_COUNT) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("attribute_kind"), value_kind,
          LOOM_BYTECODE_ATTR_COUNT, value_kind_offset);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_attr_value(
        reader, cursor, NULL, descriptor, value_kind, &attrs[attr_index]));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  uint8_t has_body = 0;
  uint64_t has_body_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &has_body));
  if (has_body > 1) {
    return loom_bytecode_reader_emit_enum_value(reader, IREE_SV("has_body"),
                                                has_body, 2, has_body_offset);
  }
  if ((vtable->region_count == 1) != (has_body != 0)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("has_body"), has_body_offset,
        IREE_SV("record_body_presence_must_match_the_defining_op_region"));
  }

  uint64_t ir_offset = 0;
  uint32_t ir_length = 0;
  if (has_body) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &ir_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(reader, cursor, &ir_length));
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("IR body"), ir_offset, ir_length,
        ir_section ? ir_section->length : 0));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  loom_op_t* op = NULL;
  uint8_t region_count = has_body ? vtable->region_count : 0;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, op_kind, 0, 0, region_count, 0, vtable->attribute_count,
      LOOM_LOCATION_NONE, &op));
  if (vtable->attribute_count > 0) {
    memcpy(loom_op_attrs(op), attrs,
           vtable->attribute_count * sizeof(loom_attribute_t));
  }
  if (has_body) {
    iree_string_view_t symbol_name = reader->strings[name_id];
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_symbol_regions(
        reader, symbol_name, ir_section, ir_offset, ir_length, builder, op,
        LOOM_REGION_INDEX_NONE, /*predefined_values=*/NULL,
        /*predefined_value_count=*/0));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, op));
  if (symbol_comment_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_op_comments(
        reader->output_module, op, symbol_comments, symbol_comment_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_materialize_symbols(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* symbols_section,
    const loom_bytecode_reader_section_t* ir_section) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_predeclare_symbols(reader, symbols_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  loom_bytecode_reader_cursor_t cursor;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_symbol_cursor_to_entries(
      reader, symbols_section, &cursor));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  loom_builder_t builder;
  loom_builder_initialize(reader->output_module, &reader->output_module->arena,
                          loom_module_block(reader->output_module), &builder);
  for (iree_host_size_t i = 0; i < reader->symbol_count; ++i) {
    uint64_t name_id = 0;
    uint8_t kind = 0;
    uint8_t visibility = 0;
    uint16_t flags = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, &cursor, &visibility));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &flags));
    (void)visibility;
    loom_string_id_t import_module_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t import_symbol_id = LOOM_STRING_ID_INVALID;
    if (flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT) {
      uint64_t encoded_import_module_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          reader, &cursor, &encoded_import_module_id));
      import_module_id = (loom_string_id_t)encoded_import_module_id;
      uint64_t encoded_import_symbol_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          reader, &cursor, &encoded_import_symbol_id));
      import_symbol_id = (loom_string_id_t)encoded_import_symbol_id;
    }
    if (kind <= LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_function_symbol(
          reader, &cursor, ir_section, name_id, kind, flags, import_module_id,
          import_symbol_id, &builder));
    } else if (kind == LOOM_BYTECODE_SYMBOL_GLOBAL) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_global_symbol(
          reader, &cursor, name_id, i, &builder));
    } else if (kind == LOOM_BYTECODE_SYMBOL_RECORD) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_record_symbol(
          reader, &cursor, ir_section, name_id, i, &builder));
    } else {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), i, IREE_SV("kind"),
          kind_offset,
          IREE_SV("module_materialization_does_not_support_this_symbol_kind"));
    }
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("SYMBOLS"));
}

static iree_status_t loom_bytecode_reader_read_symbols(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* symbols_section,
    const loom_bytecode_reader_section_t* ir_section) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      symbols_section->bytes.data, symbols_section->bytes.data_length,
      symbols_section->absolute_offset, IREE_SV("SYMBOLS"), &cursor);
  uint64_t count = 0;
  uint64_t count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (count > LOOM_BYTECODE_MAX_SYMBOL_COUNT) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("SYMBOLS"), count, LOOM_BYTECODE_MAX_SYMBOL_COUNT,
        count_offset);
  }
  reader->symbol_count = (iree_host_size_t)count;

  uint64_t import_count = 0;
  uint64_t export_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &import_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &export_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (import_count > count || export_count > count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("header"), 0,
        IREE_SV("import_export_count"),
        loom_bytecode_reader_cursor_absolute_position(&cursor),
        IREE_SV("import_export_counts_must_not_exceed_symbol_count"));
  }
  loom_bytecode_module_metadata_t* module_metadata =
      reader->current_module_metadata;
  if (module_metadata) {
    module_metadata->symbol_count = (iree_host_size_t)count;
    module_metadata->import_count = (iree_host_size_t)import_count;
    module_metadata->export_count = (iree_host_size_t)export_count;
    if (count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)count,
          sizeof(loom_bytecode_symbol_metadata_t),
          (void**)&module_metadata->symbols));
      memset(module_metadata->symbols, 0,
             (iree_host_size_t)count * sizeof(*module_metadata->symbols));
    }
    if (import_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)import_count,
          sizeof(uint32_t), (void**)&module_metadata->import_symbol_indices));
    }
    if (export_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)export_count,
          sizeof(uint32_t), (void**)&module_metadata->export_symbol_indices));
    }
  }
  uint64_t offset_table_count = import_count + export_count;
  uint64_t* offset_table = NULL;
  if (offset_table_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)offset_table_count, sizeof(uint64_t),
        (void**)&offset_table));
  }
  for (uint64_t i = 0; i < offset_table_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, &cursor, &offset_table[i]));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  uint64_t entries_base = cursor.absolute_offset + cursor.cursor.position;
  uint64_t entries_payload_length =
      symbols_section->absolute_offset + symbols_section->length - entries_base;
  for (uint64_t i = 0; i < offset_table_count; ++i) {
    if (offset_table[i] >= entries_payload_length) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("symbol_offset_table"), offset_table[i], 1,
          entries_payload_length);
    }
  }

  const uint64_t* import_offsets = offset_table;
  const uint64_t* export_offsets =
      offset_table ? offset_table + import_count : NULL;
  uint64_t import_index = 0;
  uint64_t export_index = 0;
  for (uint64_t symbol_index = 0; symbol_index < count; ++symbol_index) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t entry_relative_offset = entry_offset - entries_base;
    loom_bytecode_symbol_metadata_t* symbol_metadata =
        module_metadata ? &module_metadata->symbols[symbol_index] : NULL;
    if (symbol_metadata) {
      symbol_metadata->entry_offset = entry_offset;
    }
    uint64_t name_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    uint64_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, &name_id));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    iree_string_view_t unused_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
        reader, name_id, IREE_SV("symbol_name"), name_offset, &unused_name));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (symbol_metadata) {
      symbol_metadata->name = unused_name;
    }

    uint8_t kind = 0;
    uint8_t visibility = 0;
    uint16_t flags = 0;
    uint64_t kind_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, &cursor, &kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, &cursor, &visibility));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &flags));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (kind >= LOOM_BYTECODE_SYMBOL_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("symbol_kind"), kind, LOOM_BYTECODE_SYMBOL_COUNT_,
          kind_offset);
    }
    if (visibility >= LOOM_BYTECODE_SYMBOL_VISIBILITY_COUNT_) {
      return loom_bytecode_reader_emit_enum_value(
          reader, IREE_SV("symbol_visibility"), visibility,
          LOOM_BYTECODE_SYMBOL_VISIBILITY_COUNT_, kind_offset + 1);
    }
    if (flags &
        ~(LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC | LOOM_BYTECODE_SYMBOL_FLAG_IMPORT |
          LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("flags"), kind_offset + 2,
          IREE_SV("symbol_has_unsupported_flag_bits"));
    }
    if (iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL) &&
        !iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT)) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
          IREE_SV("flags"), kind_offset + 2,
          IREE_SV("explicit_import_symbol_flag_requires_import_flag"));
    }
    if (symbol_metadata) {
      symbol_metadata->kind = (loom_bytecode_symbol_kind_t)kind;
      symbol_metadata->visibility =
          (loom_bytecode_symbol_visibility_t)visibility;
      symbol_metadata->flags = flags;
    }
    const bool is_import =
        iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT);
    const bool is_export =
        !is_import && iree_any_bit_set(flags, LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC);
    if (is_import) {
      if (import_index >= import_count ||
          import_offsets[import_index] != entry_relative_offset) {
        uint64_t diagnostic_offset = import_index < import_count
                                         ? symbols_section->absolute_offset +
                                               import_offsets[import_index]
                                         : entry_offset;
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("import_offset_table"),
            import_index, IREE_SV("symbol_entry_offset"), diagnostic_offset,
            IREE_SV("import_offset_table_must_match_import_symbol_order"));
      }
      if (module_metadata) {
        module_metadata->import_symbol_indices[import_index] =
            (uint32_t)symbol_index;
      }
      ++import_index;
    } else if (is_export) {
      if (export_index >= export_count ||
          export_offsets[export_index] != entry_relative_offset) {
        uint64_t diagnostic_offset = export_index < export_count
                                         ? symbols_section->absolute_offset +
                                               export_offsets[export_index]
                                         : entry_offset;
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("export_offset_table"),
            export_index, IREE_SV("symbol_entry_offset"), diagnostic_offset,
            IREE_SV("export_offset_table_must_match_export_symbol_order"));
      }
      if (module_metadata) {
        module_metadata->export_symbol_indices[export_index] =
            (uint32_t)symbol_index;
      }
      ++export_index;
    }
    if (flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT) {
      uint64_t module_ref_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t module_ref_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &module_ref_id));
      iree_string_view_t import_module = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, module_ref_id, IREE_SV("source_module_id"), module_ref_offset,
          &import_module));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      uint64_t symbol_ref_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t symbol_ref_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &symbol_ref_id));
      iree_string_view_t import_symbol = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
          reader, symbol_ref_id, IREE_SV("source_symbol_id"), symbol_ref_offset,
          &import_symbol));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (symbol_metadata) {
        symbol_metadata->import_module = import_module;
        symbol_metadata->import_symbol = import_symbol;
      }
    }

    if (kind <= LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
      uint64_t op_ref_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      uint64_t op_table_index_plus1 = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          reader, &cursor, &op_table_index_plus1));
      const loom_op_vtable_t* unused_vtable = NULL;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_op_ref(
          reader, op_table_index_plus1, op_ref_offset, &unused_vtable));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      if (!unused_vtable->func_like) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
            IREE_SV("def_op_table_index_plus1"), op_ref_offset,
            IREE_SV("function_symbol_defining_op_must_implement_funclike"));
      }
      if (symbol_metadata) {
        symbol_metadata->defining_op_name = loom_op_vtable_name(unused_vtable);
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_comment_list(
          reader, &cursor, reader->arena, NULL, NULL));
      if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

      uint8_t calling_convention = 0;
      uint64_t cc_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &calling_convention));
      uint8_t purity = 0;
      uint64_t purity_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &purity));
      const loom_func_like_vtable_t* func_like = unused_vtable->func_like;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_func_enum(
          reader, symbol_index, unused_vtable, func_like->cc_attr_index,
          IREE_SV("calling_convention"), calling_convention, cc_offset));
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_func_enum(
          reader, symbol_index, unused_vtable, func_like->purity_attr_index,
          IREE_SV("purity"), purity, purity_offset));
      uint64_t arg_count = 0;
      uint64_t result_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &arg_count));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_uvarint(reader, &cursor, &result_count));
      if (arg_count > UINT16_MAX || result_count > UINT16_MAX) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
            IREE_SV("signature_count"), cc_offset,
            IREE_SV("function_signature_exceeds_runtime_field_width"));
      }
      if (symbol_metadata) {
        symbol_metadata->calling_convention = calling_convention;
        symbol_metadata->purity = purity;
        symbol_metadata->argument_count = (uint16_t)arg_count;
        symbol_metadata->result_count = (uint16_t)result_count;
      }
      for (uint64_t i = 0; i < arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_value_def(
            reader, &cursor, IREE_SV("arg_type")));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
      }
      uint64_t actual_tied_result_count = 0;
      for (uint64_t i = 0; i < result_count; ++i) {
        uint8_t is_tied = 0;
        uint64_t tie_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u8(reader, &cursor, &is_tied));
        if (is_tied > 1) {
          return loom_bytecode_reader_emit_enum_value(
              reader, IREE_SV("is_tied"), is_tied, 2, tie_offset);
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_value_def(
            reader, &cursor, IREE_SV("result_type")));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (is_tied) {
          uint64_t operand_index = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
              reader, &cursor, &operand_index));
          if (operand_index >= arg_count) {
            return loom_bytecode_reader_emit_invalid_field(
                reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
                IREE_SV("tied_operand_index"), tie_offset,
                IREE_SV("tied_result_references_an_argument_out_of_range"));
          }
          ++actual_tied_result_count;
        }
      }
      uint64_t tied_result_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
          reader, &cursor, &tied_result_count));
      if (tied_result_count != actual_tied_result_count) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
            IREE_SV("tied_result_count"),
            loom_bytecode_reader_cursor_absolute_position(&cursor),
            IREE_SV("tied_result_summary_does_not_match_tied_result_records"));
      }
      if (symbol_metadata) {
        symbol_metadata->tied_result_count = (uint16_t)tied_result_count;
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_skip_predicate_list(reader, &cursor, false));
      if (kind == LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE ||
          kind == LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL) {
        uint64_t implements_offset =
            loom_bytecode_reader_cursor_absolute_position(&cursor);
        uint64_t implements_string_id = 0;
        uint64_t priority = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_uvarint(
            reader, &cursor, &implements_string_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_uvarint(reader, &cursor, &priority));
        iree_string_view_t implements_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_string_ref(
            reader, implements_string_id, IREE_SV("implements_op_name"),
            implements_offset, &implements_name));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (symbol_metadata) {
          symbol_metadata->implements_op_name = implements_name;
          symbol_metadata->priority = priority;
        }
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_func_payload_attrs(
          reader, &cursor, symbol_index, unused_vtable, func_like));
      uint8_t has_body = 0;
      uint64_t has_body_offset =
          loom_bytecode_reader_cursor_absolute_position(&cursor);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_reader_read_u8(reader, &cursor, &has_body));
      if (has_body > 1) {
        return loom_bytecode_reader_emit_enum_value(
            reader, IREE_SV("has_body"), has_body, 2, has_body_offset);
      }
      if (has_body) {
        uint64_t ir_offset = 0;
        uint32_t ir_length = 0;
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u64_le(reader, &cursor, &ir_offset));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_reader_read_u32_le(reader, &cursor, &ir_length));
        IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
            reader, IREE_SV("IR body"), ir_offset, ir_length,
            ir_section ? ir_section->length : 0));
        if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
        if (symbol_metadata) {
          symbol_metadata->has_body = true;
          symbol_metadata->body_offset = ir_offset;
          symbol_metadata->body_absolute_offset =
              ir_section ? ir_section->absolute_offset + ir_offset : 0;
          symbol_metadata->body_length = ir_length;
        }
      }
    } else if (kind == LOOM_BYTECODE_SYMBOL_GLOBAL) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_global_payload(
          reader, &cursor, symbol_index, symbol_metadata));
    } else if (kind == LOOM_BYTECODE_SYMBOL_RECORD) {
      IREE_RETURN_IF_ERROR(loom_bytecode_reader_skip_record_payload(
          reader, &cursor, ir_section, symbol_index, symbol_metadata));
    }
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

    uint64_t next_entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor) - entries_base;
    if (next_entry_offset > entries_payload_length) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("symbol_entry"), next_entry_offset, 0,
          entries_payload_length);
    }
    if (symbol_metadata) {
      symbol_metadata->entry_length =
          loom_bytecode_reader_cursor_absolute_position(&cursor) -
          symbol_metadata->entry_offset;
    }
  }
  if (import_index != import_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("import_offset_table"),
        import_index, IREE_SV("symbol_entry_offset"),
        symbols_section->absolute_offset,
        IREE_SV("import_offset_table_contains_entries_without_import_symbols"));
  }
  if (export_index != export_count) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SYMBOLS"), IREE_SV("export_offset_table"),
        export_index, IREE_SV("symbol_entry_offset"),
        symbols_section->absolute_offset,
        IREE_SV("export_offset_table_contains_entries_without_export_symbols"));
  }
  return loom_bytecode_reader_expect_empty(reader, &cursor, IREE_SV("SYMBOLS"));
}

static iree_status_t loom_bytecode_reader_validate_file_header(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t* out_string_pool_length) {
  if (!loom_bytecode_cursor_has_bytes(&cursor->cursor, 16)) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, 0, 16, loom_bytecode_cursor_remaining(&cursor->cursor));
  }

  uint8_t magic[LOOM_BYTECODE_MAGIC_LENGTH] = {0};
  iree_const_byte_span_t magic_span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
      reader, cursor, LOOM_BYTECODE_MAGIC_LENGTH, &magic_span));
  memcpy(magic, magic_span.data, sizeof(magic));
  if (memcmp(magic, LOOM_BYTECODE_MAGIC, LOOM_BYTECODE_MAGIC_LENGTH) != 0) {
    char actual_magic[9] = {0};
    static const char kHex[] = "0123456789ABCDEF";
    for (int i = 0; i < LOOM_BYTECODE_MAGIC_LENGTH; ++i) {
      actual_magic[i * 2 + 0] = kHex[magic[i] >> 4];
      actual_magic[i * 2 + 1] = kHex[magic[i] & 0xF];
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV(LOOM_BYTECODE_MAGIC)),
        loom_param_string(iree_make_cstring_view(actual_magic)),
    };
    return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_001, params,
                                     IREE_ARRAYSIZE(params), 0,
                                     LOOM_BYTECODE_MAGIC_LENGTH);
  }

  uint8_t version = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(reader, cursor, &version));
  if (version != LOOM_BYTECODE_FORMAT_VERSION) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(version),
        loom_param_u32(LOOM_BYTECODE_FORMAT_VERSION),
    };
    return loom_bytecode_reader_emit(reader, LOOM_ERR_BYTECODE_002, params,
                                     IREE_ARRAYSIZE(params), 4, 1);
  }
  reader->format_version = version;
  uint8_t location_mode = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u8(reader, cursor, &location_mode));
  if (location_mode > LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    return loom_bytecode_reader_emit_enum_value(
        reader, IREE_SV("location_mode"), location_mode,
        LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS + 1, 5);
  }
  if (location_mode == LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("location_mode"),
        5, IREE_SV("full_locations_bytecode_requires_field_span_support"));
  }
  reader->result.location_mode = (loom_bytecode_location_mode_t)location_mode;

  uint16_t module_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u16_le(reader, cursor, &module_count));
  if (module_count == 0) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("module_count"),
        6, IREE_SV("bytecode_files_must_contain_at_least_one_module"));
  }
  reader->module_count = module_count;
  reader->result.module_count = module_count;

  uint32_t string_pool_length = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u32_le(reader, cursor, &string_pool_length));
  *out_string_pool_length = string_pool_length;
  uint32_t reserved = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_u32_le(reader, cursor, &reserved));
  if (reserved != 0) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("reserved"), 12,
        IREE_SV("reserved_header_field_must_be_zero"));
  }

  uint64_t producer_start = cursor->cursor.position;
  while (cursor->cursor.position < cursor->cursor.length &&
         cursor->cursor.data[cursor->cursor.position] != 0) {
    ++cursor->cursor.position;
  }
  if (cursor->cursor.position >= cursor->cursor.length) {
    return loom_bytecode_reader_emit_unexpected_end(
        reader, producer_start, 1,
        cursor->cursor.length >= producer_start
            ? cursor->cursor.length - producer_start
            : 0);
  }
  iree_string_view_t producer =
      iree_make_string_view((const char*)cursor->cursor.data + producer_start,
                            cursor->cursor.position - producer_start);
  if (!loom_bytecode_reader_string_is_valid_utf8(producer)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("producer"),
        producer_start, IREE_SV("producer_string_is_not_valid_utf_8"));
  }
  reader->producer = producer;
  ++cursor->cursor.position;
  while ((cursor->cursor.position & 7) != 0) {
    uint64_t padding_offset = cursor->cursor.position;
    uint8_t padding = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &padding));
    if (padding != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("FILE"), IREE_SV("header"), 0,
          IREE_SV("producer_padding"), padding_offset,
          IREE_SV("header_alignment_padding_must_be_zero"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_module_directory(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t string_pool_length) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      reader->arena, reader->module_count,
      sizeof(loom_bytecode_reader_module_t), (void**)&reader->modules));
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(cursor);
    loom_bytecode_reader_module_t* module = &reader->modules[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(reader, cursor, &module->name_offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, cursor, &module->name_length));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, cursor, &module->flags));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &module->offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, cursor, &module->length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (module->flags & ~LOOM_BYTECODE_MODULE_FLAG_DECLARATIONS_ONLY) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("FILE"), IREE_SV("module_directory"), i,
          IREE_SV("flags"), entry_offset + 6,
          IREE_SV("module_has_unsupported_flag_bits"));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("file_string_pool"), module->name_offset,
        module->name_length, string_pool_length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("module"), module->offset, module->length,
        reader->bytecode.data_length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }

  uint64_t string_pool_end = 0;
  if (string_pool_length > UINT64_MAX - cursor->cursor.position) {
    return loom_bytecode_reader_emit_range_error(
        reader, IREE_SV("file_string_pool"), cursor->cursor.position,
        string_pool_length, reader->bytecode.data_length);
  }
  string_pool_end = cursor->cursor.position + string_pool_length;
  if (string_pool_end > UINT64_MAX - 7) {
    return loom_bytecode_reader_emit_range_error(
        reader, IREE_SV("file_string_pool"), cursor->cursor.position,
        string_pool_length, reader->bytecode.data_length);
  }
  uint64_t previous_end = (string_pool_end + 7) & ~UINT64_C(7);
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    const loom_bytecode_reader_module_t* module = &reader->modules[i];
    if (module->offset < previous_end) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("module"), module->offset, module->length,
          reader->bytecode.data_length);
    }
    previous_end = module->offset + module->length;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_file_string_pool(
    loom_bytecode_reader_state_t* reader, loom_bytecode_reader_cursor_t* cursor,
    uint64_t string_pool_length) {
  iree_const_byte_span_t pool = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_span(
      reader, cursor, string_pool_length, &pool));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  reader->file_string_pool =
      iree_make_string_view((const char*)pool.data, pool.data_length);
  if (!loom_bytecode_reader_string_is_valid_utf8(reader->file_string_pool)) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("FILE"), IREE_SV("string_pool"), 0,
        IREE_SV("utf8_data"), cursor->absolute_offset,
        IREE_SV("file_string_pool_is_not_valid_utf_8"));
  }
  while ((cursor->cursor.position & 7) != 0) {
    uint64_t padding_offset = cursor->cursor.position;
    uint8_t padding = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u8(reader, cursor, &padding));
    if (padding != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("FILE"), IREE_SV("string_pool"), 0,
          IREE_SV("padding"), padding_offset,
          IREE_SV("file_string_pool_alignment_padding_must_be_zero"));
    }
  }
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    loom_bytecode_reader_module_t* module = &reader->modules[i];
    module->name = iree_make_string_view(
        reader->file_string_pool.data + module->name_offset,
        module->name_length);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_populate_file_metadata(
    loom_bytecode_reader_state_t* reader) {
  if (!reader->output_metadata) return iree_ok_status();
  loom_bytecode_file_metadata_t* metadata = reader->output_metadata;
  metadata->format_version = reader->format_version;
  metadata->location_mode = reader->result.location_mode;
  metadata->producer = reader->producer;
  metadata->module_count = reader->module_count;
  if (reader->module_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->metadata_arena, reader->module_count,
        sizeof(loom_bytecode_module_metadata_t), (void**)&metadata->modules));
    memset(metadata->modules, 0,
           reader->module_count * sizeof(loom_bytecode_module_metadata_t));
  }
  for (iree_host_size_t i = 0; i < reader->module_count; ++i) {
    const loom_bytecode_reader_module_t* source = &reader->modules[i];
    loom_bytecode_module_metadata_t* target = &metadata->modules[i];
    target->name = source->name;
    target->flags = source->flags;
    target->offset = source->offset;
    target->length = source->length;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_section_directory(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_module_t* module,
    loom_bytecode_reader_section_t** out_sections,
    iree_host_size_t* out_section_count) {
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(
      reader->bytecode.data + module->offset, (iree_host_size_t)module->length,
      module->offset, IREE_SV("MODULE"), &cursor);

  uint64_t section_count = 0;
  uint64_t section_count_offset =
      loom_bytecode_reader_cursor_absolute_position(&cursor);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(reader, &cursor, &section_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (section_count > LOOM_BYTECODE_MAX_SECTION_COUNT ||
      section_count > IREE_HOST_SIZE_MAX) {
    return loom_bytecode_reader_emit_count_exceeds(
        reader, IREE_SV("SECTIONS"), section_count,
        LOOM_BYTECODE_MAX_SECTION_COUNT, section_count_offset);
  }

  memset(&reader->current_module_summary, 0,
         sizeof(reader->current_module_summary));
  uint64_t* summary_fields[] = {
      &reader->current_module_summary.value_count,
      &reader->current_module_summary.region_count,
      &reader->current_module_summary.block_count,
      &reader->current_module_summary.op_count,
  };
  for (int i = 0; i < 4; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_uvarint(reader, &cursor, summary_fields[i]));
  }
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (module == &reader->modules[0]) {
    reader->result.first_module.value_count =
        reader->current_module_summary.value_count;
    reader->result.first_module.region_count =
        reader->current_module_summary.region_count;
    reader->result.first_module.block_count =
        reader->current_module_summary.block_count;
    reader->result.first_module.op_count =
        reader->current_module_summary.op_count;
  }

  loom_bytecode_reader_section_t* sections = NULL;
  if (section_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)section_count,
        sizeof(loom_bytecode_reader_section_t), (void**)&sections));
  }
  uint64_t minimum_section_offset = cursor.cursor.position + section_count * 32;
  uint64_t previous_end = minimum_section_offset;
  for (uint64_t i = 0; i < section_count; ++i) {
    uint64_t entry_offset =
        loom_bytecode_reader_cursor_absolute_position(&cursor);
    loom_bytecode_reader_section_t* section = &sections[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &section->kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u16_le(reader, &cursor, &section->flags));
    uint32_t reserved = 0;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u32_le(reader, &cursor, &reserved));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, &cursor, &section->offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_u64_le(reader, &cursor, &section->length));
    uint64_t uncompressed_length = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u64_le(
        reader, &cursor, &uncompressed_length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    if (section->flags != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("flags"), entry_offset + 2,
          IREE_SV("section_has_unsupported_flag_bits"));
    }
    if (reserved != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("reserved"), entry_offset + 4,
          IREE_SV("section_reserved_field_must_be_zero"));
    }
    if (uncompressed_length != 0) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
          IREE_SV("uncompressed_length"), entry_offset + 24,
          IREE_SV(
              "uncompressed length must be zero for uncompressed sections"));
    }
    for (uint64_t j = 0; j < i; ++j) {
      if (sections[j].kind == section->kind) {
        return loom_bytecode_reader_emit_invalid_field(
            reader, IREE_SV("SECTIONS"), IREE_SV("directory"), i,
            IREE_SV("kind"), entry_offset,
            IREE_SV("section_kind_appears_more_than_once"));
      }
    }
    if (section->offset < previous_end) {
      return loom_bytecode_reader_emit_range_error(
          reader, IREE_SV("section"), section->offset, section->length,
          module->length);
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_reader_validate_range(
        reader, IREE_SV("section"), section->offset, section->length,
        module->length));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
    section->absolute_offset = module->offset + section->offset;
    section->bytes = iree_make_const_byte_span(
        reader->bytecode.data + section->absolute_offset,
        (iree_host_size_t)section->length);
    previous_end = section->offset + section->length;
  }
  if (reader->current_module_metadata) {
    loom_bytecode_module_metadata_t* metadata = reader->current_module_metadata;
    metadata->summary = reader->current_module_summary;
    metadata->section_count = (iree_host_size_t)section_count;
    if (section_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          reader->metadata_arena, (iree_host_size_t)section_count,
          sizeof(loom_bytecode_section_metadata_t),
          (void**)&metadata->sections));
      for (uint64_t i = 0; i < section_count; ++i) {
        metadata->sections[i] = (loom_bytecode_section_metadata_t){
            .kind = sections[i].kind,
            .flags = sections[i].flags,
            .offset = sections[i].offset,
            .length = sections[i].length,
            .absolute_offset = sections[i].absolute_offset,
            .bytes = sections[i].bytes,
        };
      }
    }
  }
  *out_sections = sections;
  *out_section_count = (iree_host_size_t)section_count;
  return iree_ok_status();
}

static const loom_bytecode_reader_section_t* loom_bytecode_reader_find_section(
    const loom_bytecode_reader_section_t* sections, iree_host_size_t count,
    uint16_t kind) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (sections[i].kind == kind) return &sections[i];
  }
  return NULL;
}

static iree_status_t loom_bytecode_reader_require_section(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_section_t* sections, iree_host_size_t count,
    uint16_t kind, const loom_bytecode_reader_section_t** out_section) {
  const loom_bytecode_reader_section_t* section =
      loom_bytecode_reader_find_section(sections, count, kind);
  if (!section) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
        IREE_SV("required_section"), 0,
        iree_make_cstring_view(loom_bytecode_section_name(kind)));
  }
  *out_section = section;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_read_module_metadata(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_module_t* module) {
  iree_host_size_t module_ordinal =
      (iree_host_size_t)(module - reader->modules);
  reader->current_module_metadata =
      reader->output_metadata
          ? &reader->output_metadata->modules[module_ordinal]
          : NULL;
  reader->strings = NULL;
  reader->string_count = 0;
  reader->sources = NULL;
  reader->source_ids = NULL;
  reader->source_count = 0;
  reader->types = NULL;
  reader->type_count = 0;
  reader->ops = NULL;
  reader->op_kinds = NULL;
  reader->op_count = 0;
  reader->encoding_families = NULL;
  reader->encoding_family_name_ids = NULL;
  reader->encoding_family_count = 0;
  reader->encoding_count = 0;
  reader->location_count = 0;
  reader->symbol_count = 0;

  loom_bytecode_reader_section_t* sections = NULL;
  iree_host_size_t section_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_section_directory(
      reader, module, &sections, &section_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  const loom_bytecode_reader_section_t* strings_section = NULL;
  const loom_bytecode_reader_section_t* sources_section = NULL;
  const loom_bytecode_reader_section_t* encodings_section = NULL;
  const loom_bytecode_reader_section_t* types_section = NULL;
  const loom_bytecode_reader_section_t* ops_section = NULL;
  const loom_bytecode_reader_section_t* symbols_section = NULL;
  const loom_bytecode_reader_section_t* ir_section = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_STRINGS,
      &strings_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_SOURCES,
      &sources_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_ENCODINGS,
      &encodings_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_TYPES,
      &types_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_OPS,
      &ops_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_SYMBOLS,
      &symbols_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_IR, &ir_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  const loom_bytecode_reader_section_t* locations_section =
      loom_bytecode_reader_find_section(sections, section_count,
                                        LOOM_BYTECODE_SECTION_LOCATIONS);
  if (reader->result.location_mode ==
      LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    if (locations_section) {
      return loom_bytecode_reader_emit_invalid_field(
          reader, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
          IREE_SV("LOCATIONS"), locations_section->absolute_offset,
          IREE_SV("no_locations_bytecode_must_not_contain_locations"));
    }
  } else if (!locations_section) {
    return loom_bytecode_reader_emit_invalid_field(
        reader, IREE_SV("SECTIONS"), IREE_SV("directory"), 0,
        IREE_SV("LOCATIONS"), 0,
        IREE_SV("source_location_bytecode_must_contain_locations"));
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_string_table(
      reader, *strings_section, IREE_SV("STRINGS"),
      LOOM_BYTECODE_MAX_STRING_COUNT, &reader->strings, &reader->string_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_string_table(
      reader, *sources_section, IREE_SV("SOURCES"),
      LOOM_BYTECODE_MAX_STRING_COUNT, &reader->sources, &reader->source_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_encodings(reader, encodings_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_types(reader, types_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_ops(reader, ops_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (locations_section) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_read_locations(reader, locations_section));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_symbols(reader, symbols_section, ir_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  reader->current_module_summary.string_count = reader->string_count;
  reader->current_module_summary.source_count = reader->source_count;
  reader->current_module_summary.type_count = reader->type_count;
  reader->current_module_summary.encoding_count = reader->encoding_count;
  reader->current_module_summary.op_name_count = reader->op_count;
  reader->current_module_summary.location_count = reader->location_count;
  reader->current_module_summary.symbol_count = reader->symbol_count;
  if (reader->current_module_metadata) {
    reader->current_module_metadata->summary = reader->current_module_summary;
  }
  if (module == &reader->modules[0]) {
    reader->result.first_module = reader->current_module_summary;
  }
  reader->current_module_metadata = NULL;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_reader_allocate_output_module(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_module_t* module) {
  loom_module_size_hints_t hints = {
      .value_count =
          (iree_host_size_t)reader->current_module_summary.value_count,
      .string_count = reader->string_count,
      .type_count = reader->type_count,
      .symbol_count = reader->symbol_count,
  };
  // Bytecode string IDs must materialize 1:1 into module string IDs. Allocate
  // with an empty module name so STRINGS[0] can remain the value-name sentinel.
  return loom_module_allocate(reader->context, iree_string_view_empty(),
                              reader->block_pool, &hints,
                              reader->host_allocator, &reader->output_module);
}

static iree_status_t loom_bytecode_reader_materialize_module(
    loom_bytecode_reader_state_t* reader,
    const loom_bytecode_reader_module_t* module) {
  loom_bytecode_reader_section_t* sections = NULL;
  iree_host_size_t section_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_section_directory(
      reader, module, &sections, &section_count));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();

  const loom_bytecode_reader_section_t* encodings_section = NULL;
  const loom_bytecode_reader_section_t* symbols_section = NULL;
  const loom_bytecode_reader_section_t* ir_section = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_ENCODINGS,
      &encodings_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_SYMBOLS,
      &symbols_section));
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_require_section(
      reader, sections, section_count, LOOM_BYTECODE_SECTION_IR, &ir_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  const loom_bytecode_reader_section_t* locations_section =
      loom_bytecode_reader_find_section(sections, section_count,
                                        LOOM_BYTECODE_SECTION_LOCATIONS);

  IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_strings(reader));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  loom_string_id_t module_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      reader->output_module, module->name, &module_name_id));
  reader->output_module->name_id = module_name_id;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_sources(reader));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_materialize_encodings(reader, encodings_section));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_materialize_types(reader));
  if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  if (locations_section) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_reader_materialize_locations(reader, locations_section));
    if (loom_bytecode_reader_has_errors(reader)) return iree_ok_status();
  }
  return loom_bytecode_reader_materialize_symbols(reader, symbols_section,
                                                  ir_section);
}

iree_status_t loom_bytecode_read_metadata(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_bytecode_reader_state_t reader = {
      .bytecode = bytecode,
      .filename = filename,
      .context = context,
      .arena = &arena,
      .diagnostic_context =
          {
              .sink = options ? options->diagnostic_sink
                              : (loom_diagnostic_sink_t){0},
              .filename = filename,
          },
  };

  loom_bytecode_reader_cursor_t file_cursor;
  loom_bytecode_reader_cursor_initialize(bytecode.data, bytecode.data_length, 0,
                                         IREE_SV("FILE"), &file_cursor);
  uint64_t string_pool_length = 0;
  iree_status_t status = loom_bytecode_reader_validate_file_header(
      &reader, &file_cursor, &string_pool_length);
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_module_directory(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_file_string_pool(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    for (iree_host_size_t i = 0; i < reader.module_count; ++i) {
      status = loom_bytecode_reader_read_module_metadata(&reader,
                                                         &reader.modules[i]);
      if (!iree_status_is_ok(status) ||
          loom_bytecode_reader_has_errors(&reader)) {
        break;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_result = reader.result;
  }
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_bytecode_read_index(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* metadata_arena,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result,
    loom_bytecode_file_metadata_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(metadata_arena);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_ASSERT_ARGUMENT(out_metadata);
  *out_metadata = (loom_bytecode_file_metadata_t){0};

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_bytecode_reader_state_t reader = {
      .bytecode = bytecode,
      .filename = filename,
      .context = context,
      .arena = &arena,
      .metadata_arena = metadata_arena,
      .diagnostic_context =
          {
              .sink = options ? options->diagnostic_sink
                              : (loom_diagnostic_sink_t){0},
              .filename = filename,
          },
      .output_metadata = out_metadata,
  };

  loom_bytecode_reader_cursor_t file_cursor;
  loom_bytecode_reader_cursor_initialize(bytecode.data, bytecode.data_length, 0,
                                         IREE_SV("FILE"), &file_cursor);
  uint64_t string_pool_length = 0;
  iree_status_t status = loom_bytecode_reader_validate_file_header(
      &reader, &file_cursor, &string_pool_length);
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_module_directory(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_file_string_pool(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_populate_file_metadata(&reader);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    for (iree_host_size_t i = 0; i < reader.module_count; ++i) {
      status = loom_bytecode_reader_read_module_metadata(&reader,
                                                         &reader.modules[i]);
      if (!iree_status_is_ok(status) ||
          loom_bytecode_reader_has_errors(&reader)) {
        break;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_result = reader.result;
  }
  iree_arena_deinitialize(&arena);
  return status;
}

static iree_status_t loom_bytecode_read_module_impl(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    uint16_t module_ordinal, bool require_single_module,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator) {
  *out_module = NULL;

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_bytecode_reader_state_t reader = {
      .bytecode = bytecode,
      .filename = filename,
      .context = context,
      .arena = &arena,
      .diagnostic_context =
          {
              .sink = options ? options->diagnostic_sink
                              : (loom_diagnostic_sink_t){0},
              .filename = filename,
          },
      .block_pool = block_pool,
      .host_allocator = host_allocator,
  };

  loom_bytecode_reader_cursor_t file_cursor;
  loom_bytecode_reader_cursor_initialize(bytecode.data, bytecode.data_length, 0,
                                         IREE_SV("FILE"), &file_cursor);
  uint64_t string_pool_length = 0;
  iree_status_t status = loom_bytecode_reader_validate_file_header(
      &reader, &file_cursor, &string_pool_length);
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_module_directory(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_read_file_string_pool(&reader, &file_cursor,
                                                        string_pool_length);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader) &&
      require_single_module && reader.module_count != 1) {
    status = loom_bytecode_reader_emit_invalid_field(
        &reader, IREE_SV("FILE"), IREE_SV("header"), 0, IREE_SV("module_count"),
        0, IREE_SV("module_materialization_requires_exactly_one_module"));
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    if (module_ordinal >= reader.module_count) {
      status = loom_bytecode_reader_emit_invalid_field(
          &reader, IREE_SV("FILE"), IREE_SV("module_directory"), module_ordinal,
          IREE_SV("module_ordinal"), 0,
          IREE_SV("requested_module_ordinal_is_out_of_range"));
    }
  }
  loom_bytecode_reader_module_t* selected_module = NULL;
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    selected_module = &reader.modules[module_ordinal];
    status =
        loom_bytecode_reader_read_module_metadata(&reader, selected_module);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status =
        loom_bytecode_reader_allocate_output_module(&reader, selected_module);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader)) {
    status = loom_bytecode_reader_materialize_module(&reader, selected_module);
  }
  if (iree_status_is_ok(status) && !loom_bytecode_reader_has_errors(&reader) &&
      options && options->verify_module) {
    loom_verify_result_t verify_result = {0};
    loom_verify_options_t verify_options = {
        .sink = options->diagnostic_sink,
        .max_errors = options->verify_max_errors,
    };
    status = loom_verify_module(reader.output_module, &verify_options,
                                &verify_result);
    reader.result.error_count += verify_result.error_count;
    reader.result.warning_count += verify_result.warning_count;
  }

  if (iree_status_is_ok(status)) {
    *out_result = reader.result;
    if (!loom_bytecode_reader_has_errors(&reader)) {
      *out_module = reader.output_module;
      reader.output_module = NULL;
    }
  }
  if (reader.output_module) {
    loom_module_free(reader.output_module);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_bytecode_read_module(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator) {
  return loom_bytecode_read_module_impl(bytecode, filename, context, block_pool,
                                        /*module_ordinal=*/0,
                                        /*require_single_module=*/true, options,
                                        out_result, out_module, host_allocator);
}

iree_status_t loom_bytecode_read_module_ordinal(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    uint16_t module_ordinal, const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator) {
  return loom_bytecode_read_module_impl(
      bytecode, filename, context, block_pool, module_ordinal,
      /*require_single_module=*/false, options, out_result, out_module,
      host_allocator);
}
