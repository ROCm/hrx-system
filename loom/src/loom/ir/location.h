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
// Four kinds of location:
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
//   Tagged:  Compact compiler-owned metadata payload attached to an optional
//            child location. Tags are integer IDs so common tags do not touch
//            module string tables during JIT compilation.
//
// Locations can be stripped for release/embedded builds. A stripped location
// preserves the interned ID (for bytecode stability) but carries no data.
// Fusing stripped locations produces stripped.
//
// Text format:
//   loc("model.loom":42:3 to 42:58)
//   loc(fused<"model.loom":42:3, "recipe.loom":15:1>)
//   loc(opaque<"torch", "node_id=42">)
//   loc(tagged<sanitizer_site, "0102ff", "model.loom":42:3>)

#ifndef LOOM_IR_LOCATION_H_
#define LOOM_IR_LOCATION_H_

#include "iree/base/api.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Index into the module's location table.
//
// ID 0 is always LOOM_LOCATION_NONE (unknown/absent). When locations are
// stripped, all ops point to ID 0 and the location table is empty.
typedef uint32_t loom_location_id_t;
#define LOOM_LOCATION_UNKNOWN ((loom_location_id_t)0)
// Largest representable location ID. This is a valid ID; zero is the only
// reserved value.
#define LOOM_LOCATION_ID_MAX ((loom_location_id_t)UINT32_MAX)

// Index into the module's source table.
//
// The source table is a module-level table of unique source identifiers:
// filenames ("model.loom"), external system tags ("torch", "jax"), or any
// other provenance label. Cross-module cloning imports referenced source
// identifiers into the target module and remaps source IDs in copied locations.
typedef uint16_t loom_source_id_t;
#define LOOM_SOURCE_ID_INVALID ((loom_source_id_t)UINT16_MAX)

// Integer tag identifying a compact tagged location payload.
typedef uint16_t loom_location_tag_t;
// Largest valid location tag.
#define LOOM_LOCATION_TAG_MAX ((loom_location_tag_t)UINT16_MAX)

// Built-in tagged location payload tags. Values >= LOOM_LOCATION_TAG_USER_BASE
// are reserved for embedders and user-defined tooling.
typedef enum loom_location_tag_e {
  // Invalid tag sentinel. Serialized tagged locations must not use it.
  LOOM_LOCATION_TAG_INVALID = 0x0000,
  // Sanitizer assertion/reporting site metadata.
  LOOM_LOCATION_TAG_SANITIZER_SITE = 0x0001,
  // Template instantiation provenance metadata.
  LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION = 0x0002,
  // Tile dialect lowering provenance metadata.
  LOOM_LOCATION_TAG_TILE_LOWERING = 0x0003,
  // UKernel selection provenance metadata.
  LOOM_LOCATION_TAG_UKERNEL_SELECTION = 0x0004,
  // First tag value available to external users.
  LOOM_LOCATION_TAG_USER_BASE = 0x8000,
} loom_location_tag_e;

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
  // Lazy arena-owned name index. Its count identifies the indexed prefix of
  // entries; trusted unique appends are incorporated by the next registration.
  struct loom_intern_table_t* name_index;
} loom_source_table_t;

// Location kind. Stored in one byte in every location-table row.
typedef uint8_t loom_location_kind_t;
enum loom_location_kind_e {
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
  // Compiler-owned compact metadata payload attached to an optional child
  // location.
  LOOM_LOCATION_TAGGED = 4,
  LOOM_LOCATION_COUNT_,
};

// Location flags.
enum loom_location_flag_bits_e {
  // Compiler-generated op, not from user source.
  LOOM_LOCATION_FLAG_SYNTHETIC = 1u << 0,
};
typedef uint8_t loom_location_flags_t;

// Field categories for parser-captured source spans attached to file
// locations. These mirror the printer/verifier field namespaces without
// making the IR layer depend on printer- or diagnostics-specific headers.
typedef uint8_t loom_location_field_kind_t;
enum loom_location_field_kind_e {
  LOOM_LOCATION_FIELD_OPERAND = 0,
  LOOM_LOCATION_FIELD_RESULT = 1,
  LOOM_LOCATION_FIELD_ATTRIBUTE = 2,
  LOOM_LOCATION_FIELD_REGION = 3,
  LOOM_LOCATION_FIELD_SUCCESSOR = 4,
};

// Largest valid line or column in a compact field span. Coordinate zero means
// unavailable; every value through this maximum is otherwise valid.
#define LOOM_LOCATION_COMPACT_COORDINATE_MAX ((uint32_t)UINT16_MAX)

// Largest valid operation or logical field-span source coordinate.
#define LOOM_LOCATION_COORDINATE_MAX ((uint32_t)UINT32_MAX)

// Largest number of field spans retained by one file location.
#define LOOM_LOCATION_FIELD_SPAN_COUNT_MAX ((uint16_t)UINT16_MAX)

// Largest number of children retained by one fused location.
#define LOOM_LOCATION_FUSED_CHILD_COUNT_MAX ((uint32_t)UINT32_MAX)

// Largest opaque or tagged payload retained by one location.
#define LOOM_LOCATION_DATA_LENGTH_MAX ((uint32_t)UINT32_MAX)

// Physical representation used by a file location's field-span payload.
typedef uint8_t loom_location_field_span_format_t;
enum loom_location_field_span_format_e {
  // No field-span payload is attached.
  LOOM_LOCATION_FIELD_SPAN_FORMAT_NONE = 0,
  // Absolute 16-bit coordinates in loom_location_compact_field_span_t.
  LOOM_LOCATION_FIELD_SPAN_FORMAT_COMPACT = 1,
  // Absolute 32-bit coordinates in loom_location_field_span_t.
  LOOM_LOCATION_FIELD_SPAN_FORMAT_WIDE = 2,
};

// Logical source span for one concrete op field inside a file location. Lines
// and Unicode code-point columns are one-based; the end is exclusive. This is
// also the physical wide representation.
typedef struct loom_location_field_span_t {
  // Field category interpreted by the parser, printer, and diagnostics.
  loom_location_field_kind_t kind;
  // Reserved for representation growth; must be zero.
  uint8_t reserved;
  // Index within the field category.
  uint16_t index;
  // One-based inclusive start line, or zero when unavailable.
  uint32_t start_line;
  // One-based inclusive start Unicode code-point column, or zero.
  uint32_t start_col;
  // One-based exclusive end line, or zero when unavailable.
  uint32_t end_line;
  // One-based exclusive end Unicode code-point column, or zero.
  uint32_t end_col;
} loom_location_field_span_t;

static_assert(sizeof(loom_location_field_span_t) == 20,
              "wide location field spans must be 20 bytes");

// Common physical field-span representation. Coordinates are absolute and
// have the same semantics as loom_location_field_span_t.
typedef struct loom_location_compact_field_span_t {
  // Field category interpreted by the parser, printer, and diagnostics.
  loom_location_field_kind_t kind;
  // Reserved for representation growth; must be zero.
  uint8_t reserved;
  // Index within the field category.
  uint16_t index;
  // One-based inclusive start line, or zero when unavailable.
  uint16_t start_line;
  // One-based inclusive start Unicode code-point column, or zero.
  uint16_t start_col;
  // One-based exclusive end line, or zero when unavailable.
  uint16_t end_line;
  // One-based exclusive end Unicode code-point column, or zero.
  uint16_t end_col;
} loom_location_compact_field_span_t;

static_assert(sizeof(loom_location_compact_field_span_t) == 12,
              "compact location field spans must be 12 bytes");

// Borrowed physical field-span array supplied for module attachment.
typedef struct loom_location_field_span_array_t {
  // Physical representation used by entries in |data|.
  loom_location_field_span_format_t format;
  // Number of physical spans in |data|.
  iree_host_size_t count;
  // Borrowed compact or wide span storage selected by |format|.
  const void* data;
} loom_location_field_span_array_t;

// Returns an empty field-span array.
static inline loom_location_field_span_array_t
loom_location_field_span_array_empty(void) {
  return (loom_location_field_span_array_t){
      .format = LOOM_LOCATION_FIELD_SPAN_FORMAT_NONE,
  };
}

// Returns a borrowed compact field-span array.
static inline loom_location_field_span_array_t
loom_location_field_span_array_compact(
    const loom_location_compact_field_span_t* spans, iree_host_size_t count) {
  return (loom_location_field_span_array_t){
      .format = LOOM_LOCATION_FIELD_SPAN_FORMAT_COMPACT,
      .count = count,
      .data = spans,
  };
}

// Returns a borrowed wide field-span array.
static inline loom_location_field_span_array_t
loom_location_field_span_array_wide(const loom_location_field_span_t* spans,
                                    iree_host_size_t count) {
  return (loom_location_field_span_array_t){
      .format = LOOM_LOCATION_FIELD_SPAN_FORMAT_WIDE,
      .count = count,
      .data = spans,
  };
}

// A compact, pointer-width-dependent source location entry. Tagged union.
//
// The kind field determines which union variant is active. File location rows
// retain full uint32_t operation coordinates. Their field spans independently
// select compact 16-bit or wide 32-bit physical storage per operation.
//
// File lines and Unicode code-point columns are one-based, with an exclusive
// end; zero denotes an unavailable coordinate. The source_id field indexes into
// the module's source table, which stores filenames, system tags, and any other
// provenance labels.
typedef union loom_location_entry_t {
  // Common prefix shared by every active variant.
  struct {
    // Active loom_location_kind_e variant.
    loom_location_kind_t kind;
    // Common loom_location_flag_bits_e flags.
    loom_location_flags_t flags;
  };

  // LOOM_LOCATION_FILE: source range.
  struct {
    // Active kind; must be LOOM_LOCATION_FILE.
    loom_location_kind_t kind;
    // Common location flags.
    loom_location_flags_t flags;
    // Module source-table identifier.
    loom_source_id_t source_id;
    // Number of spans in |field_span_data|.
    uint16_t field_span_count;
    // Physical representation of |field_span_data|.
    loom_location_field_span_format_t field_span_format;
    // Reserved for representation growth; must be zero.
    uint8_t reserved;
    // One-based inclusive operation start line, or zero when unavailable.
    uint32_t start_line;
    // One-based inclusive operation start Unicode code-point column, or zero.
    uint32_t start_col;
    // One-based exclusive operation end line, or zero when unavailable.
    uint32_t end_line;
    // One-based exclusive operation end Unicode code-point column, or zero.
    uint32_t end_col;
    // Module-owned compact or wide spans selected by |field_span_format|.
    const void* field_span_data;
  } file;

  // LOOM_LOCATION_FUSED: multiple source locations.
  struct {
    // Active kind; must be LOOM_LOCATION_FUSED.
    loom_location_kind_t kind;
    // Common location flags.
    loom_location_flags_t flags;
    // Reserved for representation growth; must be zero.
    uint16_t reserved;
    // Number of location IDs in |children|.
    uint32_t count;
    // Module-owned child location IDs.
    loom_location_id_t* children;
  } fused;

  // LOOM_LOCATION_OPAQUE: external identifier.
  struct {
    // Active kind; must be LOOM_LOCATION_OPAQUE.
    loom_location_kind_t kind;
    // Common location flags.
    loom_location_flags_t flags;
    // Module source-table identifier naming the external system.
    loom_source_id_t source_id;
    // Byte length of |data|.
    uint32_t data_length;
    // Module-owned opaque identifier bytes.
    const uint8_t* data;
  } opaque;

  // LOOM_LOCATION_TAGGED: compact metadata payload with optional child.
  struct {
    // Active kind; must be LOOM_LOCATION_TAGGED.
    loom_location_kind_t kind;
    // Common location flags.
    loom_location_flags_t flags;
    // Integer tag identifying the payload schema.
    loom_location_tag_t tag;
    // Byte length of the payload pointed to by data.
    uint32_t data_length;
    // Optional child location preserving source or provenance context.
    loom_location_id_t child;
    // Borrowed payload bytes owned by the containing module arena.
    const uint8_t* data;
  } tagged;
} loom_location_entry_t;

static_assert(sizeof(loom_location_entry_t) == (IREE_PTR_SIZE == 8 ? 32 : 28),
              "loom_location_entry_t must remain compact");

// Number of rows in one stable location segment.
#define LOOM_LOCATION_SEGMENT_CAPACITY 256u

// Shift mapping a location ID to its segment index.
#define LOOM_LOCATION_SEGMENT_SHIFT 8u

// Mask mapping a location ID to its row within a segment.
#define LOOM_LOCATION_SEGMENT_MASK (LOOM_LOCATION_SEGMENT_CAPACITY - 1u)

static_assert((1u << LOOM_LOCATION_SEGMENT_SHIFT) ==
                  LOOM_LOCATION_SEGMENT_CAPACITY,
              "location segment capacity must match its index shift");
static_assert((uint64_t)LOOM_LOCATION_SEGMENT_CAPACITY *
                      LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT >=
                  (uint64_t)LOOM_LOCATION_ID_MAX + 1,
              "location storage must cover the full location ID domain");

// Fixed-size arena-owned location rows; appending never moves earlier rows.
typedef struct loom_location_segment_t {
  // Rows indexed by the low bits of a location ID.
  loom_location_entry_t rows[LOOM_LOCATION_SEGMENT_CAPACITY];
} loom_location_segment_t;

// Location table stored on the module.
//
// Entry 0 is always LOOM_LOCATION_NONE. When locations are stripped, the table
// is empty and all ops reference LOOM_LOCATION_UNKNOWN. Rows are allocated
// lazily in stable segments and remain valid until module destruction.
typedef struct loom_location_table_t {
  // Number of initialized rows, including the unknown entry when nonempty.
  iree_host_size_t count;
  // Stable row segments owned by the containing module's arena.
  loom_segmented_storage_t segments;
} loom_location_table_t;

// Returns a mutable published row for |location_id|, which must be < count.
static inline loom_location_entry_t* loom_location_table_entry(
    loom_location_table_t* table, loom_location_id_t location_id) {
  IREE_ASSERT((iree_host_size_t)location_id < table->count);
  loom_location_segment_t* segment =
      (loom_location_segment_t*)loom_segmented_storage_segment(
          &table->segments, location_id >> LOOM_LOCATION_SEGMENT_SHIFT);
  return &segment->rows[location_id & LOOM_LOCATION_SEGMENT_MASK];
}

// Returns a const published row for |location_id|, which must be < count.
static inline const loom_location_entry_t* loom_location_table_const_entry(
    const loom_location_table_t* table, loom_location_id_t location_id) {
  IREE_ASSERT((iree_host_size_t)location_id < table->count);
  const loom_location_segment_t* segment =
      (const loom_location_segment_t*)loom_segmented_storage_const_segment(
          &table->segments, location_id >> LOOM_LOCATION_SEGMENT_SHIFT);
  return &segment->rows[location_id & LOOM_LOCATION_SEGMENT_MASK];
}

// Returns the kind tag for |entry|.
loom_location_kind_t loom_location_get_kind(loom_location_entry_t entry);

// Returns the number of bytes occupied by |count| spans in |format|. The
// format must be compact or wide when |count| is nonzero.
iree_host_size_t loom_location_field_span_data_length(
    loom_location_field_span_format_t format, iree_host_size_t count);

// Returns field span |index| normalized to absolute uint32_t coordinates.
// |entry| must be a file location and |index| must be in range.
loom_location_field_span_t loom_location_file_field_span_at(
    const loom_location_entry_t* entry, uint16_t index);

// Constructs a file source range location.
loom_location_entry_t loom_location_file_range(loom_source_id_t source_id,
                                               uint32_t start_line,
                                               uint32_t start_col,
                                               uint32_t end_line,
                                               uint32_t end_col);

// Constructs a tagged metadata location. The data payload is borrowed from the
// caller and must outlive the containing module entry.
loom_location_entry_t loom_location_tagged(loom_location_tag_t tag,
                                           loom_location_id_t child,
                                           const uint8_t* data,
                                           uint32_t data_length);

// Returns the canonical textual name for a built-in location tag.
iree_string_view_t loom_location_tag_name(loom_location_tag_t tag);

// Parses a canonical built-in location tag name.
bool loom_location_tag_parse(iree_string_view_t name,
                             loom_location_tag_t* out_tag);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_LOCATION_H_
