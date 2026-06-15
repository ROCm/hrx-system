// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/varint.h"

iree_status_t loom_uvarint_decode(loom_bytecode_cursor_t* cursor,
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
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "uvarint overflow at offset %" PRIhsz
                              ": 10th byte is 0x%02x (must be 0 or 1)",
                              start_position, byte);
    }

    value |= (uint64_t)payload << shift;

    if ((byte & 0x80) == 0) {
      if (shift > 0 && payload == 0) {
        cursor->position = start_position;
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "non-canonical uvarint at offset %" PRIhsz
                                ": final byte carries no payload bits",
                                start_position);
      }
      // Continuation bit clear: varint complete.
      *out_value = value;
      return iree_ok_status();
    }

    shift += 7;

    // After 10 bytes (shift would be 70), the varint is too long.
    if (shift > 63) {
      cursor->position = start_position;
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "uvarint too long at offset %" PRIhsz
                              ": exceeds 10-byte maximum",
                              start_position);
    }
  }

  // Ran out of data with continuation bit still set.
  cursor->position = start_position;
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "truncated uvarint at offset %" PRIhsz
                          ": continuation bit set at end of data",
                          start_position);
}

iree_status_t loom_svarint_decode(loom_bytecode_cursor_t* cursor,
                                  int64_t* out_value) {
  uint64_t zigzag = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_decode(cursor, &zigzag));
  // Zigzag decode: inverse of (value << 1) ^ (value >> 63).
  *out_value = (int64_t)((zigzag >> 1) ^ -(zigzag & 1));
  return iree_ok_status();
}
