// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/indexed_source.h"

// Index admission established canonical LEB128 encodings, their bounds, and
// coordinate limits. This cold query consumes those facts without revalidation.
static uint32_t loom_bytecode_index_read_coordinate(const uint8_t** cursor) {
  uint32_t value = 0;
  unsigned shift = 0;
  uint8_t byte;
  do {
    byte = *(*cursor)++;
    value |= (uint32_t)(byte & 0x7F) << shift;
    shift += 7;
  } while (byte & 0x80);
  return value;
}

bool loom_bytecode_index_resolve_source(
    iree_const_byte_span_t bytecode,
    const loom_bytecode_module_metadata_t* metadata,
    loom_location_id_t location, loom_source_range_t* out_range) {
  while (location != LOOM_LOCATION_UNKNOWN) {
    const uint8_t* cursor =
        bytecode.data + metadata->locations.entries[location].entry_offset;
    const loom_location_kind_t kind = (loom_location_kind_t)cursor[0];
    cursor += 2;  // Kind and validated flags.
    if (kind == LOOM_LOCATION_FILE) {
      const uint32_t source_id = loom_bytecode_index_read_coordinate(&cursor);
      *out_range = (loom_source_range_t){
          .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
          .filename = metadata->sources.values[source_id],
      };
      out_range->start_line = loom_bytecode_index_read_coordinate(&cursor);
      out_range->start_column = loom_bytecode_index_read_coordinate(&cursor);
      out_range->end_line = loom_bytecode_index_read_coordinate(&cursor);
      out_range->end_column = loom_bytecode_index_read_coordinate(&cursor);
      return true;
    }
    if (kind != LOOM_LOCATION_TAGGED) {
      return false;
    }
    loom_bytecode_index_read_coordinate(&cursor);  // Tag.
    location = loom_bytecode_index_read_coordinate(&cursor);
  }
  return false;
}
