// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source location model for Loom IR.
//
// Every IR operation carries a location for diagnostics, source mapping, and
// debug info generation. Locations are interned per module and referenced by a
// 4-byte loom_location_id_t on each operation.
//
// Three kinds of location:
//
//   File:    Source range from a .loom file. Stores file:start:end so tooling
//            can extract exact text spans for diagnostics, diffs, and
//            agent-driven code modification. Parser-created file locations may
//            also carry per-field token spans for source-backed verifier
//            highlights.
//   Fused:   Derived from multiple source locations when a pass creates an op
//            from several inputs (inlining, fusion, rewrites).
//   Opaque:  External system identifier for JIT (torch node ID, JAX trace
//            position). Round-trips through compilation.
//
// Locations can be stripped for release/embedded builds. A stripped location
// preserves the interned ID (for bytecode stability) but carries no data.
// Fusing stripped locations produces stripped.
//
// Text format:
//   loc("model.loom":42:3 to 42:58)
//   loc(fused<"model.loom":42:3, "recipe.loom":15:1>)
//   loc(opaque<"torch", "node_id=42">)

#ifndef LOOM_IR_LOCATION_H_
#define LOOM_IR_LOCATION_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Index into the module's location table.
//
// ID 0 is always LOOM_LOCATION_NONE (unknown/absent). When locations are
// stripped, all ops point to ID 0 and the location table is empty.
typedef uint32_t loom_location_id_t;
#define LOOM_LOCATION_UNKNOWN ((loom_location_id_t)0)

// Index into the module's source table.
//
// The source table is a module-level table of unique source identifiers:
// filenames ("model.loom"), external system tags ("torch", "jax"), or any
// other provenance label. Cross-module cloning imports referenced source
// identifiers into the target module and remaps source IDs in copied locations.
typedef uint16_t loom_source_id_t;
#define LOOM_SOURCE_ID_INVALID ((loom_source_id_t)UINT16_MAX)

// Module-owned source table. Entries are interned copies of source names
// (filenames, system tags). The table is append-only and deduplicates by exact
// string match.
typedef struct loom_source_table_t {
  // Number of source identifiers stored in entries.
  iree_host_size_t count;
  // Allocated entry capacity.
  iree_host_size_t capacity;
  // Module-arena-owned source identifier entries.
  iree_string_view_t* entries;
} loom_source_table_t;

// Location kind.
typedef enum loom_location_kind_e {
  // No location information. Synthetic or stripped.
  LOOM_LOCATION_NONE = 0,
  // Source range from a .loom file: file + start + end positions.
  LOOM_LOCATION_FILE = 1,
  // Derived from multiple source locations (inlining, fusion).
  // Children are arena-allocated location IDs.
  LOOM_LOCATION_FUSED = 2,
  // External system identifier (torch node ID, JAX trace).
  // Tag + opaque data blob, round-tripped verbatim.
  LOOM_LOCATION_OPAQUE = 3,
  LOOM_LOCATION_COUNT_,
} loom_location_kind_t;

// Location flags.
enum loom_location_flag_bits_e {
  // Compiler-generated op, not from user source.
  LOOM_LOCATION_FLAG_SYNTHETIC = 1u << 0,
};
typedef uint8_t loom_location_flags_t;

// Field categories for parser-captured source spans attached to file
// locations. These mirror the printer/verifier field namespaces without
// making the IR layer depend on printer- or diagnostics-specific headers.
typedef enum loom_location_field_kind_e {
  LOOM_LOCATION_FIELD_OPERAND = 0,
  LOOM_LOCATION_FIELD_RESULT = 1,
  LOOM_LOCATION_FIELD_ATTRIBUTE = 2,
  LOOM_LOCATION_FIELD_REGION = 3,
  LOOM_LOCATION_FIELD_SUCCESSOR = 4,
} loom_location_field_kind_t;

// Source span for one concrete op field inside a file location.
typedef struct loom_location_field_span_t {
  loom_location_field_kind_t kind;
  uint16_t index;
  uint16_t start_line;
  uint16_t start_col;
  uint16_t end_line;
  uint16_t end_col;
} loom_location_field_span_t;

static_assert(sizeof(loom_location_field_span_t) == 16,
              "loom_location_field_span_t must be 16 bytes");

// A source location entry. 24 bytes. Tagged union.
//
// The kind field determines which union variant is active. File locations (the
// 90% case) use uint16_t line/column numbers, supporting up to 65K lines per
// file. Agent-authored .loom files are typically hundreds of lines; linked
// module dumps are diagnostic output, not round-tripped source.
//
// The source_id field indexes into the module's source table, which stores
// filenames, system tags, and any other provenance labels.
typedef struct loom_location_entry_t {
  loom_location_kind_t kind;
  loom_location_flags_t flags;
  union {
    // LOOM_LOCATION_FILE: source range.
    struct {
      loom_source_id_t source_id;
      uint16_t field_span_count;
      uint16_t start_line;
      uint16_t start_col;
      uint16_t end_line;
      uint16_t end_col;
      const loom_location_field_span_t* field_spans;
    } file;

    // LOOM_LOCATION_FUSED: multiple source locations.
    struct {
      uint32_t count;
      loom_location_id_t* children;
    } fused;

    // LOOM_LOCATION_OPAQUE: external identifier.
    struct {
      loom_source_id_t source_id;
      uint32_t data_length;
      const uint8_t* data;
    } opaque;
  };
} loom_location_entry_t;

static_assert(sizeof(loom_location_entry_t) == 32,
              "loom_location_entry_t must be 32 bytes");

// Location table stored on the module.
//
// Entry 0 is always LOOM_LOCATION_NONE. When locations are stripped, the table
// is empty and all ops reference LOOM_LOCATION_UNKNOWN.
typedef struct loom_location_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_location_entry_t* entries;
} loom_location_table_t;

// Returns the kind tag for |entry|.
loom_location_kind_t loom_location_get_kind(loom_location_entry_t entry);

// Constructs a file source range location.
loom_location_entry_t loom_location_file_range(loom_source_id_t source_id,
                                               uint16_t start_line,
                                               uint16_t start_col,
                                               uint16_t end_line,
                                               uint16_t end_col);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_LOCATION_H_
