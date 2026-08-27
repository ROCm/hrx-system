// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/source_trivia.h"

#include "iree/base/internal/unicode.h"
#include "loom/format/bytecode/format.h"

#define LOOM_BYTECODE_MAX_SOURCE_COMMENT_LENGTH (UINT64_C(1) << 24)

static iree_status_t loom_bytecode_source_trivia_read_header(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, bool* out_leading_blank_line,
    uint16_t* out_comment_count) {
  const uint64_t source_trivia_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t source_trivia = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &source_trivia));
  const uint64_t comment_count =
      source_trivia >> LOOM_BYTECODE_SOURCE_TRIVIA_COMMENT_COUNT_SHIFT;
  if (comment_count > UINT16_MAX) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("comment_list"), 0,
        IREE_SV("comment_count"), source_trivia_offset,
        IREE_SV("comment_count_exceeds_field_width"));
  }
  *out_leading_blank_line =
      (source_trivia & LOOM_BYTECODE_SOURCE_TRIVIA_LEADING_BLANK_LINE) != 0;
  *out_comment_count = (uint16_t)comment_count;
  return iree_ok_status();
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_status_t
loom_bytecode_source_trivia_read_comment(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint16_t comment_index,
    iree_string_view_t* out_comment) {
  const uint64_t length_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint64_t length = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_uvarint(decoder, cursor, &length));
  if (length > LOOM_BYTECODE_MAX_SOURCE_COMMENT_LENGTH) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("comment_list"), comment_index,
        IREE_SV("comment_length"), length_offset,
        IREE_SV("comment_length_exceeds_maximum"));
  }
  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      loom_bytecode_reader_read_span(decoder, cursor, length, &span));
  *out_comment =
      iree_make_string_view((const char*)span.data, span.data_length);
  if (!iree_unicode_utf8_validate(*out_comment)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, cursor->range_name, IREE_SV("comment_list"), comment_index,
        IREE_SV("comment"),
        loom_bytecode_reader_cursor_absolute_position(cursor) - length,
        IREE_SV("comment_is_not_valid_utf_8"));
  }
  if (iree_string_view_starts_with(*out_comment, IREE_SV(" "))) {
    *out_comment = iree_string_view_remove_prefix(*out_comment, 1);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_source_trivia_validate(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor) {
  bool leading_blank_line = false;
  uint16_t comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_read_header(
      decoder, cursor, &leading_blank_line, &comment_count));
  for (uint16_t i = 0; i < comment_count; ++i) {
    iree_string_view_t comment = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_bytecode_source_trivia_read_comment(decoder, cursor, i, &comment));
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_source_trivia_materialize(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, iree_arena_allocator_t* arena,
    loom_bytecode_source_trivia_t* out_source_trivia) {
  bool leading_blank_line = false;
  uint16_t comment_count = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_read_header(
      decoder, cursor, &leading_blank_line, &comment_count));

  iree_string_view_t* comments = NULL;
  if (comment_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, comment_count, sizeof(*comments), (void**)&comments));
  }
  for (uint16_t i = 0; i < comment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_source_trivia_read_comment(
        decoder, cursor, i, &comments[i]));
  }

  *out_source_trivia = (loom_bytecode_source_trivia_t){
      .comments = comments,
      .comment_count = comment_count,
      .leading_blank_line = leading_blank_line,
  };
  return iree_ok_status();
}
