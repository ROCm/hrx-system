// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded bytecode decoding with structured malformed-input diagnostics.

#ifndef LOOM_FORMAT_BYTECODE_READER_DECODER_H_
#define LOOM_FORMAT_BYTECODE_READER_DECODER_H_

#include "iree/base/api.h"
#include "loom/error/error_defs.h"
#include "loom/format/bytecode/diagnostic.h"
#include "loom/format/bytecode/varint.h"

#ifdef __cplusplus
extern "C" {
#endif

// Diagnostic state shared by bounded bytecode cursors in one public read.
//
// The decoder owns no input, arena, table, context, or IR state. Error and
// warning counters remain owned by the caller's public result object.
typedef struct loom_bytecode_reader_decoder_t {
  // Structured diagnostic destination and input identity.
  loom_bytecode_reader_diagnostic_context_t diagnostic_context;
  // Caller-owned diagnostic counters updated by emitted diagnostics.
  struct {
    // Number of emitted error diagnostics.
    uint32_t* error;
    // Number of emitted warning diagnostics.
    uint32_t* warning;
  } counts;
} loom_bytecode_reader_decoder_t;

// Initializes a bounded decoder over caller-owned diagnostic state.
static inline void loom_bytecode_reader_decoder_initialize(
    loom_diagnostic_sink_t sink, iree_string_view_t filename,
    uint32_t* error_count, uint32_t* warning_count,
    loom_bytecode_reader_decoder_t* out_decoder) {
  *out_decoder = (loom_bytecode_reader_decoder_t){
      .diagnostic_context =
          {
              .sink = sink,
              .filename = filename,
          },
      .counts =
          {
              .error = error_count,
              .warning = warning_count,
          },
  };
}

// Returns true after the decoder has emitted an error diagnostic.
static inline bool loom_bytecode_reader_has_errors(
    const loom_bytecode_reader_decoder_t* decoder) {
  return *decoder->counts.error > 0;
}

// Bounded cursor over one named bytecode range.
typedef struct loom_bytecode_reader_cursor_t {
  // Raw bounded cursor over the range bytes.
  loom_bytecode_cursor_t cursor;
  // Absolute file offset corresponding to cursor byte zero.
  uint64_t absolute_offset;
  // Human-readable range name used in diagnostics.
  iree_string_view_t range_name;
} loom_bytecode_reader_cursor_t;

// Initializes a bounded cursor over one bytecode range.
static inline void loom_bytecode_reader_cursor_initialize(
    const uint8_t* data, iree_host_size_t length, uint64_t absolute_offset,
    iree_string_view_t range_name, loom_bytecode_reader_cursor_t* out_cursor) {
  loom_bytecode_cursor_initialize(data, length, &out_cursor->cursor);
  out_cursor->absolute_offset = absolute_offset;
  out_cursor->range_name = range_name;
}

// Returns the cursor position as an absolute file byte offset.
static inline uint64_t loom_bytecode_reader_cursor_absolute_position(
    const loom_bytecode_reader_cursor_t* cursor) {
  return cursor->absolute_offset + (uint64_t)cursor->cursor.position;
}

// Emits a structured bytecode diagnostic and updates decoder counters.
iree_status_t loom_bytecode_reader_emit(loom_bytecode_reader_decoder_t* decoder,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count,
                                        uint64_t offset, uint64_t length);

// Emits a truncated-range diagnostic at |offset|.
iree_status_t loom_bytecode_reader_emit_unexpected_end(
    loom_bytecode_reader_decoder_t* decoder, uint64_t offset, uint64_t needed,
    uint64_t available);

// Consumes a failed raw varint status and emits its structured diagnostic.
iree_status_t loom_bytecode_reader_emit_invalid_varint(
    loom_bytecode_reader_decoder_t* decoder, uint64_t offset,
    iree_status_t decode_status);

// Emits an invalid table-field diagnostic.
iree_status_t loom_bytecode_reader_emit_invalid_field(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t section_name,
    iree_string_view_t table_name, uint64_t record_index,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t failure_code);

// Emits an invalid byte-range diagnostic.
iree_status_t loom_bytecode_reader_emit_range_error(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t range_name,
    uint64_t offset, uint64_t length, uint64_t container_length);

// Emits a table-count limit diagnostic.
iree_status_t loom_bytecode_reader_emit_count_exceeds(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t table_name,
    uint64_t count, uint64_t limit, uint64_t offset);

// Emits an out-of-range table-reference diagnostic.
iree_status_t loom_bytecode_reader_emit_table_ref(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t table_name,
    uint64_t reference_id, uint64_t table_count, uint64_t offset);

// Emits an out-of-range dense-enum diagnostic.
iree_status_t loom_bytecode_reader_emit_enum_value(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t field_name,
    uint64_t actual_value, uint64_t case_count, uint64_t offset);

// Reads an unsigned byte from |cursor|.
static inline iree_status_t loom_bytecode_reader_read_u8(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint8_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(&cursor->cursor, 1))) {
    return loom_bytecode_reader_emit_unexpected_end(
        decoder, loom_bytecode_reader_cursor_absolute_position(cursor), 1,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u8(&cursor->cursor, out_value);
}

// Reads a little-endian unsigned 16-bit integer from |cursor|.
static inline iree_status_t loom_bytecode_reader_read_u16_le(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint16_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(&cursor->cursor, 2))) {
    return loom_bytecode_reader_emit_unexpected_end(
        decoder, loom_bytecode_reader_cursor_absolute_position(cursor), 2,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u16_le(&cursor->cursor, out_value);
}

// Reads a little-endian unsigned 32-bit integer from |cursor|.
static inline iree_status_t loom_bytecode_reader_read_u32_le(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint32_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(&cursor->cursor, 4))) {
    return loom_bytecode_reader_emit_unexpected_end(
        decoder, loom_bytecode_reader_cursor_absolute_position(cursor), 4,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u32_le(&cursor->cursor, out_value);
}

// Reads a little-endian unsigned 64-bit integer from |cursor|.
static inline iree_status_t loom_bytecode_reader_read_u64_le(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(&cursor->cursor, 8))) {
    return loom_bytecode_reader_emit_unexpected_end(
        decoder, loom_bytecode_reader_cursor_absolute_position(cursor), 8,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_u64_le(&cursor->cursor, out_value);
}

// Reads a canonical unsigned LEB128 integer from |cursor|.
static inline iree_status_t loom_bytecode_reader_read_uvarint(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t* out_value) {
  const uint64_t offset = loom_bytecode_reader_cursor_absolute_position(cursor);
  iree_status_t status = loom_uvarint_decode(&cursor->cursor, out_value);
  if (IREE_LIKELY(iree_status_is_ok(status))) return iree_ok_status();
  return loom_bytecode_reader_emit_invalid_varint(decoder, offset, status);
}

// Reads a canonical zigzag-encoded signed LEB128 integer from |cursor|.
static inline iree_status_t loom_bytecode_reader_read_svarint(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, int64_t* out_value) {
  uint64_t zigzag = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &zigzag));
  if (loom_bytecode_reader_has_errors(decoder)) return iree_ok_status();
  *out_value = (int64_t)((zigzag >> 1) ^ -(zigzag & 1));
  return iree_ok_status();
}

// Reads a borrowed byte span from |cursor|.
static inline iree_status_t loom_bytecode_reader_read_span(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t length,
    iree_const_byte_span_t* out_span) {
  if (IREE_UNLIKELY(length > (uint64_t)IREE_HOST_SIZE_MAX)) {
    return loom_bytecode_reader_emit_range_error(
        decoder, cursor->range_name,
        loom_bytecode_reader_cursor_absolute_position(cursor), length,
        cursor->absolute_offset + cursor->cursor.length);
  }
  const iree_host_size_t host_length = (iree_host_size_t)length;
  if (IREE_UNLIKELY(
          !loom_bytecode_cursor_has_bytes(&cursor->cursor, host_length))) {
    return loom_bytecode_reader_emit_unexpected_end(
        decoder, loom_bytecode_reader_cursor_absolute_position(cursor), length,
        loom_bytecode_cursor_remaining(&cursor->cursor));
  }
  return loom_bytecode_cursor_read_span(&cursor->cursor, host_length, out_span);
}

// Requires that |cursor| has no unread trailing bytes.
static inline iree_status_t loom_bytecode_reader_expect_empty(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, iree_string_view_t table_name) {
  if (IREE_LIKELY(loom_bytecode_cursor_is_empty(&cursor->cursor))) {
    return iree_ok_status();
  }
  return loom_bytecode_reader_emit_invalid_field(
      decoder, cursor->range_name, table_name, 0, IREE_SV("trailing_bytes"),
      loom_bytecode_reader_cursor_absolute_position(cursor),
      IREE_SV("section_has_unread_trailing_bytes"));
}

// Validates a relative byte range against its containing byte length.
static inline iree_status_t loom_bytecode_reader_validate_range(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t range_name,
    uint64_t offset, uint64_t length, uint64_t container_length) {
  if (IREE_UNLIKELY(length > UINT64_MAX - offset ||
                    offset + length > container_length)) {
    return loom_bytecode_reader_emit_range_error(decoder, range_name, offset,
                                                 length, container_length);
  }
  return iree_ok_status();
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_DECODER_H_
