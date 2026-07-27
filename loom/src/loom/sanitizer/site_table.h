// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Binary sanitizer site tables.
//
// Site tables are compact read-only blobs emitted with executable artifacts.
// Runtime reports carry dense site IDs; host diagnostics use this table to map
// those IDs back to site kind, source span, and optional sanitizer payload
// metadata. The table is useful but not required for execution: stripped builds
// may report unknown source metadata while preserving distinct site IDs.

#ifndef LOOM_SANITIZER_SITE_TABLE_H_
#define LOOM_SANITIZER_SITE_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/sanitizer/site_collection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Default symbol name used when embedding sanitizer site tables in artifacts.
#define LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME "loom_sanitizer_sites"

// Little-endian magic bytes "LSIT".
#define LOOM_SANITIZER_SITE_TABLE_MAGIC 0x5449534Cu

// Current sanitizer site table format version.
#define LOOM_SANITIZER_SITE_TABLE_VERSION 1u

// Byte length of the fixed table header.
#define LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH 32u

// Byte length of one fixed table record.
#define LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH 48u

enum loom_sanitizer_site_table_header_offset_e {
  LOOM_SANITIZER_SITE_TABLE_HEADER_MAGIC_OFFSET = 0,
  LOOM_SANITIZER_SITE_TABLE_HEADER_VERSION_OFFSET = 4,
  LOOM_SANITIZER_SITE_TABLE_HEADER_HEADER_LENGTH_OFFSET = 5,
  LOOM_SANITIZER_SITE_TABLE_HEADER_RECORD_LENGTH_OFFSET = 6,
  LOOM_SANITIZER_SITE_TABLE_HEADER_ROW_COUNT_OFFSET = 8,
  LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_OFFSET_OFFSET = 12,
  LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_LENGTH_OFFSET = 16,
  LOOM_SANITIZER_SITE_TABLE_HEADER_PAYLOAD_DATA_OFFSET_OFFSET = 20,
  LOOM_SANITIZER_SITE_TABLE_HEADER_PAYLOAD_DATA_LENGTH_OFFSET = 24,
  LOOM_SANITIZER_SITE_TABLE_HEADER_FLAGS_OFFSET = 28,
};

enum loom_sanitizer_site_table_record_offset_e {
  LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET = 0,
  LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET = 4,
  LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET = 8,
  LOOM_SANITIZER_SITE_TABLE_RECORD_PAYLOAD_OFFSET_OFFSET = 12,
  LOOM_SANITIZER_SITE_TABLE_RECORD_PAYLOAD_LENGTH_OFFSET = 16,
  LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET = 20,
  LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET = 24,
  LOOM_SANITIZER_SITE_TABLE_RECORD_START_LINE_OFFSET = 28,
  LOOM_SANITIZER_SITE_TABLE_RECORD_START_COLUMN_OFFSET = 32,
  LOOM_SANITIZER_SITE_TABLE_RECORD_END_LINE_OFFSET = 36,
  LOOM_SANITIZER_SITE_TABLE_RECORD_END_COLUMN_OFFSET = 40,
  LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET = 44,
  LOOM_SANITIZER_SITE_TABLE_RECORD_RESERVED_OFFSET = 46,
};

enum loom_sanitizer_site_table_record_flag_bits_e {
  // Record has encoded sanitizer site payload bytes.
  LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_PAYLOAD = 1u << 0,
  // Record has a file source location.
  LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION = 1u << 1,
};

// Bitfield of loom_sanitizer_site_table_record_flag_bits_e values.
typedef uint32_t loom_sanitizer_site_table_record_flags_t;

enum loom_sanitizer_site_table_source_kind_e {
  // Record has no source location.
  LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_NONE = 0,
  // Record source fields identify a file source range.
  LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_FILE = 1,
};

// Two-byte source kind stored in each sanitizer site table record.
typedef uint16_t loom_sanitizer_site_table_source_kind_t;

// Encodes |collection| into an arena-owned binary site table.
//
// Source names are deduplicated in a trailing NUL-terminated string table.
// Payloads are re-encoded using the current sanitizer site payload format into
// a trailing payload byte pool. Records for unknown or stripped locations are
// still emitted with source kind NONE so report site IDs remain distinct.
iree_status_t loom_sanitizer_site_table_encode(
    const loom_module_t* module,
    const loom_sanitizer_site_collection_t* collection,
    iree_arena_allocator_t* arena, iree_const_byte_span_t* out_table);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_SANITIZER_SITE_TABLE_H_
