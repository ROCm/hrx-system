// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/encoder.h"

#include <string.h>

#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/varint.h"
#include "loom/ir/module.h"

//===----------------------------------------------------------------------===//
// Page-buffered stream writer
//===----------------------------------------------------------------------===//
//
// Amortizes iree_io_stream_write vtable dispatch cost by accumulating
// small writes in a local 4KB page buffer. Flushes to the stream when
// full. All section data flows through this writer.

void loom_bytecode_page_writer_initialize(loom_bytecode_page_writer_t* writer,
                                          iree_io_stream_t* stream) {
  writer->stream = stream;
  writer->position = 0;
  writer->total_written = 0;
}

iree_status_t loom_bytecode_page_writer_flush(
    loom_bytecode_page_writer_t* writer) {
  if (writer->position == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(writer->stream, writer->position, writer->page));
  writer->position = 0;
  return iree_ok_status();
}

iree_status_t loom_bytecode_page_writer_write(
    loom_bytecode_page_writer_t* writer, const void* data,
    iree_host_size_t length) {
  if (length == 0) return iree_ok_status();
  writer->total_written += length;
  const uint8_t* source = (const uint8_t*)data;

  // Fast path: fits in current page.
  if (IREE_LIKELY(writer->position + length <=
                  LOOM_BYTECODE_WRITER_PAGE_SIZE)) {
    memcpy(writer->page + writer->position, source, length);
    writer->position += length;
    return iree_ok_status();
  }

  // Fill the remainder of the current page.
  iree_host_size_t remaining =
      LOOM_BYTECODE_WRITER_PAGE_SIZE - writer->position;
  if (remaining > 0) {
    memcpy(writer->page + writer->position, source, remaining);
    writer->position = LOOM_BYTECODE_WRITER_PAGE_SIZE;
    source += remaining;
    length -= remaining;
  }

  // Flush the full page.
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_flush(writer));

  // Write full pages directly to the stream (skip the page buffer).
  while (length >= LOOM_BYTECODE_WRITER_PAGE_SIZE) {
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        writer->stream, LOOM_BYTECODE_WRITER_PAGE_SIZE, source));
    source += LOOM_BYTECODE_WRITER_PAGE_SIZE;
    length -= LOOM_BYTECODE_WRITER_PAGE_SIZE;
  }

  // Copy remainder into the empty page.
  if (length > 0) {
    memcpy(writer->page, source, length);
    writer->position = length;
  }
  return iree_ok_status();
}

// Typed write helpers. Each encodes into a small stack buffer, then
// writes through the page writer.

iree_status_t loom_bytecode_page_writer_write_u8(
    loom_bytecode_page_writer_t* writer, uint8_t value) {
  return loom_bytecode_page_writer_write(writer, &value, 1);
}

iree_status_t loom_bytecode_page_writer_write_u16_le(
    loom_bytecode_page_writer_t* writer, uint16_t value) {
  uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
  return loom_bytecode_page_writer_write(writer, bytes, 2);
}

iree_status_t loom_bytecode_page_writer_write_u32_le(
    loom_bytecode_page_writer_t* writer, uint32_t value) {
  uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8),
                      (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
  return loom_bytecode_page_writer_write(writer, bytes, 4);
}

iree_status_t loom_bytecode_page_writer_write_u64_le(
    loom_bytecode_page_writer_t* writer, uint64_t value) {
  uint8_t bytes[8] = {
      (uint8_t)value,         (uint8_t)(value >> 8),  (uint8_t)(value >> 16),
      (uint8_t)(value >> 24), (uint8_t)(value >> 32), (uint8_t)(value >> 40),
      (uint8_t)(value >> 48), (uint8_t)(value >> 56),
  };
  return loom_bytecode_page_writer_write(writer, bytes, 8);
}

iree_status_t loom_bytecode_page_writer_write_uvarint(
    loom_bytecode_page_writer_t* writer, uint64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_encode(value, span, &length));
  return loom_bytecode_page_writer_write(writer, buffer, length);
}

iree_status_t loom_bytecode_page_writer_write_svarint(
    loom_bytecode_page_writer_t* writer, int64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_svarint_encode(value, span, &length));
  return loom_bytecode_page_writer_write(writer, buffer, length);
}

iree_status_t loom_bytecode_page_writer_write_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, text.size));
  return loom_bytecode_page_writer_write(writer, text.data, text.size);
}

iree_status_t loom_bytecode_page_writer_write_zeros(
    loom_bytecode_page_writer_t* writer, iree_host_size_t count) {
  static const uint8_t zeros[64] = {0};
  while (count > 0) {
    iree_host_size_t chunk = count < sizeof(zeros) ? count : sizeof(zeros);
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, zeros, chunk));
    count -= chunk;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_page_writer_pad_to_alignment(
    loom_bytecode_page_writer_t* writer, iree_host_size_t alignment) {
  iree_host_size_t remainder = writer->total_written % alignment;
  if (remainder == 0) return iree_ok_status();
  return loom_bytecode_page_writer_write_zeros(writer, alignment - remainder);
}

// Writes a null-terminated string followed by padding to 8-byte alignment.
iree_status_t loom_bytecode_page_writer_write_null_terminated_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write(writer, text.data, text.size));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 0));
  return iree_ok_status();
}

static iree_status_t loom_bytecode_encode_source_trivia(
    bool leading_blank_line, iree_host_size_t comment_count,
    uint64_t* out_source_trivia) {
  if (comment_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "comment count %" PRIhsz " exceeds maximum %u",
                            comment_count, (unsigned)UINT16_MAX);
  }
  *out_source_trivia = (uint64_t)comment_count
                       << LOOM_BYTECODE_SOURCE_TRIVIA_COMMENT_COUNT_SHIFT;
  if (leading_blank_line) {
    *out_source_trivia |= LOOM_BYTECODE_SOURCE_TRIVIA_LEADING_BLANK_LINE;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_page_writer_write_source_trivia(
    loom_bytecode_page_writer_t* writer, bool leading_blank_line,
    const iree_string_view_t* comments, iree_host_size_t comment_count) {
  uint64_t source_trivia = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_encode_source_trivia(
      leading_blank_line, comment_count, &source_trivia));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, source_trivia));
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    bool has_payload = !iree_string_view_is_empty(comments[i]);
    iree_host_size_t wire_size = comments[i].size + (has_payload ? 1 : 0);
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, wire_size));
    if (has_payload) {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, ' '));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
        writer, comments[i].data, comments[i].size));
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_source_trivia_section(
    loom_bytecode_page_writer_t* writer, const loom_module_t* module) {
  iree_host_size_t line_count = 0;
  const iree_string_view_t* lines =
      loom_module_file_header(module, &line_count);
  return loom_bytecode_page_writer_write_source_trivia(
      writer, /*leading_blank_line=*/false, lines, line_count);
}

//===----------------------------------------------------------------------===//
// String builder emit helpers (for SYMBOLS section buffering)
//===----------------------------------------------------------------------===//

iree_status_t loom_bytecode_emit_u8(iree_string_builder_t* builder,
                                    uint8_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 1, &head));
  if (head) head[0] = (char)value;
  return iree_ok_status();
}

iree_status_t loom_bytecode_emit_u16_le(iree_string_builder_t* builder,
                                        uint16_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 2, &head));
  if (head) {
    head[0] = (char)(value & 0xFF);
    head[1] = (char)((value >> 8) & 0xFF);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_emit_u32_le(iree_string_builder_t* builder,
                                        uint32_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 4, &head));
  if (head) {
    head[0] = (char)(value & 0xFF);
    head[1] = (char)((value >> 8) & 0xFF);
    head[2] = (char)((value >> 16) & 0xFF);
    head[3] = (char)((value >> 24) & 0xFF);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_emit_u64_le(iree_string_builder_t* builder,
                                        uint64_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 8, &head));
  if (head) {
    for (int i = 0; i < 8; ++i) {
      head[i] = (char)((value >> (i * 8)) & 0xFF);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_emit_uvarint(iree_string_builder_t* builder,
                                         uint64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_encode(value, span, &length));
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)buffer, length));
}

iree_status_t loom_bytecode_emit_svarint(iree_string_builder_t* builder,
                                         int64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_svarint_encode(value, span, &length));
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)buffer, length));
}

void loom_bytecode_patch_u64_le(iree_string_builder_t* builder,
                                iree_host_size_t offset, uint64_t value) {
  char* buffer = builder->buffer;
  for (int i = 0; i < 8; ++i) {
    buffer[offset + i] = (char)((value >> (i * 8)) & 0xFF);
  }
}

iree_status_t loom_bytecode_emit_source_trivia(
    iree_string_builder_t* builder, bool leading_blank_line,
    const iree_string_view_t* comments, iree_host_size_t comment_count) {
  uint64_t source_trivia = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_encode_source_trivia(
      leading_blank_line, comment_count, &source_trivia));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, source_trivia));
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    bool has_payload = !iree_string_view_is_empty(comments[i]);
    iree_host_size_t wire_size = comments[i].size + (has_payload ? 1 : 0);
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, wire_size));
    if (has_payload) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(builder, IREE_SV(" ")));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, comments[i]));
  }
  return iree_ok_status();
}
