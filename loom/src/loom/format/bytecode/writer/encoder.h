// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Primitive bytecode encoding into streams and buffered section payloads.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_ENCODER_H_
#define LOOM_FORMAT_BYTECODE_WRITER_ENCODER_H_

#include "iree/base/api.h"
#include "iree/io/stream.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bytes accumulated before one stream vtable write.
#define LOOM_BYTECODE_WRITER_PAGE_SIZE 4096

// Page-buffered output shared by all streaming bytecode sections.
typedef struct loom_bytecode_page_writer_t {
  // Destination stream receiving flushed pages.
  iree_io_stream_t* stream;
  // Pending bytes not yet flushed to the destination stream.
  uint8_t page[LOOM_BYTECODE_WRITER_PAGE_SIZE];
  // Number of pending bytes in |page|.
  iree_host_size_t position;
  // Logical byte count including flushed and pending bytes.
  uint64_t total_written;
} loom_bytecode_page_writer_t;

// Initializes |writer| to append to |stream| from logical offset zero.
void loom_bytecode_page_writer_initialize(loom_bytecode_page_writer_t* writer,
                                          iree_io_stream_t* stream);

// Flushes pending bytes to the destination stream.
iree_status_t loom_bytecode_page_writer_flush(
    loom_bytecode_page_writer_t* writer);

// Appends raw bytes to the buffered destination stream.
iree_status_t loom_bytecode_page_writer_write(
    loom_bytecode_page_writer_t* writer, const void* data,
    iree_host_size_t length);

// Appends fixed-width little-endian integers.
iree_status_t loom_bytecode_page_writer_write_u8(
    loom_bytecode_page_writer_t* writer, uint8_t value);
iree_status_t loom_bytecode_page_writer_write_u16_le(
    loom_bytecode_page_writer_t* writer, uint16_t value);
iree_status_t loom_bytecode_page_writer_write_u32_le(
    loom_bytecode_page_writer_t* writer, uint32_t value);
iree_status_t loom_bytecode_page_writer_write_u64_le(
    loom_bytecode_page_writer_t* writer, uint64_t value);

// Appends canonical unsigned or signed variable-width integers.
iree_status_t loom_bytecode_page_writer_write_uvarint(
    loom_bytecode_page_writer_t* writer, uint64_t value);
iree_status_t loom_bytecode_page_writer_write_svarint(
    loom_bytecode_page_writer_t* writer, int64_t value);

// Appends a length-prefixed string.
iree_status_t loom_bytecode_page_writer_write_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text);

// Appends zero bytes or advances to the requested byte alignment.
iree_status_t loom_bytecode_page_writer_write_zeros(
    loom_bytecode_page_writer_t* writer, iree_host_size_t count);
iree_status_t loom_bytecode_page_writer_pad_to_alignment(
    loom_bytecode_page_writer_t* writer, iree_host_size_t alignment);

// Appends a null-terminated string.
iree_status_t loom_bytecode_page_writer_write_null_terminated_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text);

// Appends the source presentation attached to one IR entity.
iree_status_t loom_bytecode_page_writer_write_source_trivia(
    loom_bytecode_page_writer_t* writer, bool leading_blank_line,
    const iree_string_view_t* comments, iree_host_size_t comment_count);

// Appends the module-level SOURCE_TRIVIA section.
iree_status_t loom_bytecode_write_source_trivia_section(
    loom_bytecode_page_writer_t* writer, const loom_module_t* module);

// Appends fixed-width little-endian integers to a buffered section payload.
iree_status_t loom_bytecode_emit_u8(iree_string_builder_t* builder,
                                    uint8_t value);
iree_status_t loom_bytecode_emit_u16_le(iree_string_builder_t* builder,
                                        uint16_t value);
iree_status_t loom_bytecode_emit_u32_le(iree_string_builder_t* builder,
                                        uint32_t value);
iree_status_t loom_bytecode_emit_u64_le(iree_string_builder_t* builder,
                                        uint64_t value);

// Appends canonical unsigned or signed variable-width integers to a buffered
// section payload.
iree_status_t loom_bytecode_emit_uvarint(iree_string_builder_t* builder,
                                         uint64_t value);
iree_status_t loom_bytecode_emit_svarint(iree_string_builder_t* builder,
                                         int64_t value);

// Patches an existing fixed-width little-endian integer in |builder|.
void loom_bytecode_patch_u64_le(iree_string_builder_t* builder,
                                iree_host_size_t offset, uint64_t value);

// Appends source presentation to a buffered section payload.
iree_status_t loom_bytecode_emit_source_trivia(
    iree_string_builder_t* builder, bool leading_blank_line,
    const iree_string_view_t* comments, iree_host_size_t comment_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_ENCODER_H_
