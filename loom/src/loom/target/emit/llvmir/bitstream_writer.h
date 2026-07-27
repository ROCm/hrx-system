// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM bitstream writer.
//
// LLVM bitcode is built from a little-endian bitstream: fields are appended
// least-significant bit first, VBR records are packed at arbitrary bit offsets,
// and block bodies align to 32-bit boundaries. This writer owns only that
// physical stream contract. LLVM IR records, abbreviations, type tables,
// constants, and function bodies live above it.

#ifndef LOOM_TARGET_LLVMIR_BITSTREAM_WRITER_H_
#define LOOM_TARGET_LLVMIR_BITSTREAM_WRITER_H_

#include "iree/io/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LLVMIR_BITSTREAM_PAGE_SIZE 4096

typedef struct loom_llvmir_bitstream_writer_t {
  // Destination stream for completed bytes.
  iree_io_stream_t* stream;
  // Absolute stream byte offset corresponding to bitstream bit offset zero.
  iree_io_stream_pos_t base_offset;
  // Buffered complete bytes waiting for the next stream write.
  uint8_t page[LOOM_LLVMIR_BITSTREAM_PAGE_SIZE];
  // Number of complete bytes currently stored in |page|.
  iree_host_size_t page_position;
  // Pending low-order bits not yet forming a complete byte.
  uint64_t pending_bits;
  // Number of valid low-order bits in |pending_bits|.
  uint32_t pending_bit_count;
  // Physical bit offset including alignment and final byte padding.
  uint64_t bit_offset;
} loom_llvmir_bitstream_writer_t;

// Initializes |out_writer| to append to |stream|. The stream must outlive the
// writer and must not be written directly until the writer is finished.
void loom_llvmir_bitstream_writer_initialize(
    iree_io_stream_t* stream, loom_llvmir_bitstream_writer_t* out_writer);

// Returns the current physical bit offset.
uint64_t loom_llvmir_bitstream_writer_bit_offset(
    const loom_llvmir_bitstream_writer_t* writer);

// Writes the low |bit_count| bits of |value| to the stream. |bit_count| may be
// zero. Non-zero high bits outside |bit_count| are rejected.
iree_status_t loom_llvmir_bitstream_writer_write_bits(
    loom_llvmir_bitstream_writer_t* writer, uint64_t value, uint32_t bit_count);

// Writes an LLVM VBR field using groups of |width| bits. The high bit of each
// group is the continuation flag; the remaining bits carry payload.
iree_status_t loom_llvmir_bitstream_writer_write_vbr(
    loom_llvmir_bitstream_writer_t* writer, uint64_t value, uint32_t width);

// Writes zero bits until the stream is aligned to a 32-bit boundary.
iree_status_t loom_llvmir_bitstream_writer_align32(
    loom_llvmir_bitstream_writer_t* writer);

// Writes a byte payload. If the bitstream is byte-aligned this goes through the
// page buffer directly; otherwise each byte is appended as an 8-bit field.
iree_status_t loom_llvmir_bitstream_writer_write_bytes(
    loom_llvmir_bitstream_writer_t* writer, const void* data,
    iree_host_size_t length);

// Patches a little-endian u32 at |bit_offset|. The offset must be byte-aligned,
// the writer must not have pending partial-byte state, and the destination
// stream must be seekable.
iree_status_t loom_llvmir_bitstream_writer_patch_u32(
    loom_llvmir_bitstream_writer_t* writer, uint64_t bit_offset,
    uint32_t value);

// Flushes complete bytes to the destination stream while preserving any pending
// partial byte in the writer.
iree_status_t loom_llvmir_bitstream_writer_flush(
    loom_llvmir_bitstream_writer_t* writer);

// Pads the current byte with zero bits and flushes all completed bytes.
iree_status_t loom_llvmir_bitstream_writer_finish(
    loom_llvmir_bitstream_writer_t* writer);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_BITSTREAM_WRITER_H_
