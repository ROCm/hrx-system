// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/varint.h"

loom_varint_decode_result_t loom_uvarint_decode(loom_bytecode_cursor_t* cursor,
                                                uint64_t* out_value) {
  iree_host_size_t start_position = cursor->position;
  uint64_t value = 0;
  uint32_t shift = 0;

  while (cursor->position < cursor->length) {
    uint8_t byte = cursor->data[cursor->position];
    ++cursor->position;

    // Check for overflow before incorporating this byte's data bits.
    // The 10th byte (shift == 63) may only carry 1 data bit (bit 0).
    uint8_t payload = byte & 0x7F;
    if (shift == 63 && payload > 1) {
      cursor->position = start_position;
      return LOOM_VARINT_DECODE_INVALID;
    }

    value |= (uint64_t)payload << shift;

    if ((byte & 0x80) == 0) {
      if (shift > 0 && payload == 0) {
        cursor->position = start_position;
        return LOOM_VARINT_DECODE_INVALID;
      }
      // Continuation bit clear: varint complete.
      *out_value = value;
      return LOOM_VARINT_DECODE_COMPLETE;
    }

    shift += 7;

    // After 10 bytes (shift would be 70), the varint is too long.
    if (shift > 63) {
      cursor->position = start_position;
      return LOOM_VARINT_DECODE_INVALID;
    }
  }

  // Ran out of data with continuation bit still set.
  cursor->position = start_position;
  return LOOM_VARINT_DECODE_TRUNCATED;
}

loom_varint_decode_result_t loom_svarint_decode(loom_bytecode_cursor_t* cursor,
                                                int64_t* out_value) {
  uint64_t zigzag = 0;
  const loom_varint_decode_result_t result =
      loom_uvarint_decode(cursor, &zigzag);
  if (IREE_UNLIKELY(result != LOOM_VARINT_DECODE_COMPLETE)) {
    return result;
  }
  // Zigzag decode: inverse of (value << 1) ^ (value >> 63).
  *out_value = (int64_t)((zigzag >> 1) ^ -(zigzag & 1));
  return LOOM_VARINT_DECODE_COMPLETE;
}
