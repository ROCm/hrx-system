// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LEB128 variable-length integer encoding for the loom bytecode format.
//
// Unsigned varints use LEB128 (Little-Endian Base 128), the same encoding
// used by DWARF, protobuf, and MLIR bytecode. Each byte stores 7 data bits
// in bits 0-6 and a continuation flag in bit 7:
//
//   value 0-127:       1 byte  (bit 7 clear)
//   value 128-16383:   2 bytes (first byte has bit 7 set)
//   ...up to 10 bytes for uint64.
//
// Signed varints use zigzag encoding before LEB128:
//   encoded = (value << 1) ^ (value >> 63)
//
// This maps small-magnitude values to small unsigned values:
//   0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
//
// ==========================================================================
// Security model
// ==========================================================================
//
// All decode operations are bounds-checked through loom_bytecode_cursor_t:
// the cursor tracks data, length, and position as a single unit. No read
// can exceed the cursor's bounds. Malformed input produces clear errors:
//   - Truncated varints (continuation bit set at end of data).
//   - Non-canonical varints (extra zero payload groups).
//   - Overflow (decoded value exceeds uint64 range).
//
// Encode operations write into bounded spans (iree_byte_span_t) and
// validate capacity before writing. The writer's page buffer uses these
// internally with pre-checked capacity for zero-overhead hot paths.

#ifndef LOOM_FORMAT_BYTECODE_VARINT_H_
#define LOOM_FORMAT_BYTECODE_VARINT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of bytes a LEB128-encoded uint64 can occupy.
// ceil(64 / 7) = 10 bytes. The 10th byte carries only 1 data bit.
#define LOOM_VARINT_MAX_LENGTH 10

//===----------------------------------------------------------------------===//
// Bytecode cursor
//===----------------------------------------------------------------------===//

// Bounded read cursor over a contiguous byte range. All decode operations
// read through the cursor and cannot exceed its bounds. The cursor advances
// on successful reads and remains unchanged on failure.
//
// Initialize from a byte span:
//   loom_bytecode_cursor_t cursor;
//   loom_bytecode_cursor_initialize(data, length, &cursor);
//
// Or from an offset within an existing span:
//   loom_bytecode_cursor_t cursor;
//   loom_bytecode_cursor_initialize(data + offset, section_length, &cursor);
typedef struct loom_bytecode_cursor_t {
  const uint8_t* data;        // Base pointer (never modified after init).
  iree_host_size_t length;    // Total byte count of the range.
  iree_host_size_t position;  // Current read position within [0, length].
} loom_bytecode_cursor_t;

// Initializes a cursor over the byte range [data, data + length).
static inline void loom_bytecode_cursor_initialize(
    const uint8_t* data, iree_host_size_t length,
    loom_bytecode_cursor_t* out_cursor) {
  out_cursor->data = data;
  out_cursor->length = length;
  out_cursor->position = 0;
}

// Returns the number of bytes remaining from the current position.
static inline iree_host_size_t loom_bytecode_cursor_remaining(
    const loom_bytecode_cursor_t* cursor) {
  if (IREE_UNLIKELY(cursor->position > cursor->length)) return 0;
  return cursor->length - cursor->position;
}

// Returns true if |byte_count| bytes can be read without exceeding bounds.
static inline bool loom_bytecode_cursor_has_bytes(
    const loom_bytecode_cursor_t* cursor, iree_host_size_t byte_count) {
  return cursor->position <= cursor->length &&
         byte_count <= cursor->length - cursor->position;
}

// Returns true if the cursor has been fully consumed.
static inline bool loom_bytecode_cursor_is_empty(
    const loom_bytecode_cursor_t* cursor) {
  return cursor->position >= cursor->length;
}

// Reads a single byte from the cursor, advancing the position.
static inline iree_status_t loom_bytecode_cursor_read_u8(
    loom_bytecode_cursor_t* cursor, uint8_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(cursor, 1))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "unexpected end of bytecode data at offset %" PRIhsz, cursor->position);
  }
  *out_value = cursor->data[cursor->position++];
  return iree_ok_status();
}

// Reads a little-endian uint16 from the cursor, advancing the position.
static inline iree_status_t loom_bytecode_cursor_read_u16_le(
    loom_bytecode_cursor_t* cursor, uint16_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(cursor, 2))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "need 2 bytes for u16 at offset %" PRIhsz
                            " but only %" PRIhsz " remain",
                            cursor->position,
                            loom_bytecode_cursor_remaining(cursor));
  }
  const uint8_t* p = cursor->data + cursor->position;
  *out_value = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  cursor->position += 2;
  return iree_ok_status();
}

// Reads a little-endian uint32 from the cursor, advancing the position.
static inline iree_status_t loom_bytecode_cursor_read_u32_le(
    loom_bytecode_cursor_t* cursor, uint32_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(cursor, 4))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "need 4 bytes for u32 at offset %" PRIhsz
                            " but only %" PRIhsz " remain",
                            cursor->position,
                            loom_bytecode_cursor_remaining(cursor));
  }
  const uint8_t* p = cursor->data + cursor->position;
  *out_value = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
               ((uint32_t)p[3] << 24);
  cursor->position += 4;
  return iree_ok_status();
}

// Reads a little-endian uint64 from the cursor, advancing the position.
static inline iree_status_t loom_bytecode_cursor_read_u64_le(
    loom_bytecode_cursor_t* cursor, uint64_t* out_value) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(cursor, 8))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "need 8 bytes for u64 at offset %" PRIhsz
                            " but only %" PRIhsz " remain",
                            cursor->position,
                            loom_bytecode_cursor_remaining(cursor));
  }
  const uint8_t* p = cursor->data + cursor->position;
  *out_value = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
               ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) |
               ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) |
               ((uint64_t)p[7] << 56);
  cursor->position += 8;
  return iree_ok_status();
}

// Reads |byte_count| raw bytes from the cursor into |out_data|.
// |out_data| must have room for |byte_count| bytes.
static inline iree_status_t loom_bytecode_cursor_read_bytes(
    loom_bytecode_cursor_t* cursor, iree_host_size_t byte_count,
    uint8_t* out_data) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(cursor, byte_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "need %" PRIhsz " bytes at offset %" PRIhsz
                            " but only %" PRIhsz " remain",
                            byte_count, cursor->position,
                            loom_bytecode_cursor_remaining(cursor));
  }
  memcpy(out_data, cursor->data + cursor->position, byte_count);
  cursor->position += byte_count;
  return iree_ok_status();
}

// Returns a const pointer to |byte_count| bytes at the current position
// and advances the cursor. The returned pointer is valid for the lifetime
// of the underlying data. No copy is made.
static inline iree_status_t loom_bytecode_cursor_read_span(
    loom_bytecode_cursor_t* cursor, iree_host_size_t byte_count,
    iree_const_byte_span_t* out_span) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(cursor, byte_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "need %" PRIhsz " bytes at offset %" PRIhsz
                            " but only %" PRIhsz " remain",
                            byte_count, cursor->position,
                            loom_bytecode_cursor_remaining(cursor));
  }
  out_span->data = cursor->data + cursor->position;
  out_span->data_length = byte_count;
  cursor->position += byte_count;
  return iree_ok_status();
}

// Advances the cursor position by |byte_count| without reading.
static inline iree_status_t loom_bytecode_cursor_skip(
    loom_bytecode_cursor_t* cursor, iree_host_size_t byte_count) {
  if (IREE_UNLIKELY(!loom_bytecode_cursor_has_bytes(cursor, byte_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "cannot skip %" PRIhsz " bytes at offset %" PRIhsz
                            " with only %" PRIhsz " remaining",
                            byte_count, cursor->position,
                            loom_bytecode_cursor_remaining(cursor));
  }
  cursor->position += byte_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Varint length computation
//===----------------------------------------------------------------------===//

// Returns the number of bytes needed to LEB128-encode |value|.
static inline iree_host_size_t loom_uvarint_length(uint64_t value) {
  iree_host_size_t length = 1;
  while (value >= 0x80) {
    value >>= 7;
    ++length;
  }
  return length;
}

// Returns the number of bytes needed to zigzag + LEB128 encode |value|.
static inline iree_host_size_t loom_svarint_length(int64_t value) {
  uint64_t zigzag = ((uint64_t)value << 1) ^ (uint64_t)(value >> 63);
  return loom_uvarint_length(zigzag);
}

//===----------------------------------------------------------------------===//
// Varint encoding (write path)
//===----------------------------------------------------------------------===//

// Encodes an unsigned 64-bit integer as LEB128 into |buffer|.
// Returns the number of bytes written through |out_length|.
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the buffer is too small.
static inline iree_status_t loom_uvarint_encode(uint64_t value,
                                                iree_byte_span_t buffer,
                                                iree_host_size_t* out_length) {
  iree_host_size_t needed = loom_uvarint_length(value);
  if (IREE_UNLIKELY(buffer.data_length < needed)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "uvarint encoding needs %" PRIhsz
                            " bytes but buffer has %" PRIhsz,
                            needed, buffer.data_length);
  }
  iree_host_size_t i = 0;
  do {
    uint8_t byte = (uint8_t)(value & 0x7F);
    value >>= 7;
    if (value != 0) byte |= 0x80;
    buffer.data[i++] = byte;
  } while (value != 0);
  *out_length = i;
  return iree_ok_status();
}

// Encodes a signed 64-bit integer using zigzag + LEB128 into |buffer|.
// Returns the number of bytes written through |out_length|.
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the buffer is too small.
static inline iree_status_t loom_svarint_encode(int64_t value,
                                                iree_byte_span_t buffer,
                                                iree_host_size_t* out_length) {
  uint64_t zigzag = ((uint64_t)value << 1) ^ (uint64_t)(value >> 63);
  return loom_uvarint_encode(zigzag, buffer, out_length);
}

//===----------------------------------------------------------------------===//
// Varint decoding (read path)
//===----------------------------------------------------------------------===//

// Decodes an unsigned LEB128 varint from |cursor|.
// Advances the cursor past the varint on success. On failure, the cursor
// position is unchanged.
//
// Returns IREE_STATUS_OUT_OF_RANGE if:
//   - The cursor has insufficient data (truncated varint).
// Returns IREE_STATUS_INVALID_ARGUMENT if:
//   - The varint uses a non-canonical overlong representation.
//   - The varint overflows uint64 (>10 bytes or 10th byte > 1).
iree_status_t loom_uvarint_decode(loom_bytecode_cursor_t* cursor,
                                  uint64_t* out_value);

// Decodes a signed zigzag + LEB128 varint from |cursor|.
// Advances the cursor past the varint on success. On failure, the cursor
// position is unchanged.
iree_status_t loom_svarint_decode(loom_bytecode_cursor_t* cursor,
                                  int64_t* out_value);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_VARINT_H_
