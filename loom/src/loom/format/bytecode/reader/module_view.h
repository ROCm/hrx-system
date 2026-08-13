// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scratch-lived facts established while validating one bytecode module.

#ifndef LOOM_FORMAT_BYTECODE_READER_MODULE_VIEW_H_
#define LOOM_FORMAT_BYTECODE_READER_MODULE_VIEW_H_

#include "iree/base/api.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/module_summary.h"
#include "loom/format/bytecode/reader/source_trivia.h"
#include "loom/format/bytecode/reader/type_plan.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// One validated module-directory entry.
typedef struct loom_bytecode_reader_module_t {
  // Offset into the file string pool.
  uint32_t name_offset;
  // Module name byte length.
  uint16_t name_length;
  // Module flags from the directory entry.
  uint16_t flags;
  // Absolute module byte offset.
  uint64_t offset;
  // Module byte length.
  uint64_t length;
  // Name view into the file string pool.
  iree_string_view_t name;
} loom_bytecode_reader_module_t;

// One validated module section-directory entry.
typedef struct loom_bytecode_reader_section_t {
  // Wire section kind.
  uint16_t kind;
  // Section flags.
  uint16_t flags;
  // Module-relative byte offset.
  uint64_t offset;
  // Section byte length.
  uint64_t length;
  // Absolute file byte offset.
  uint64_t absolute_offset;
  // Section payload bytes.
  iree_const_byte_span_t bytes;
} loom_bytecode_reader_section_t;

// One validated compile-time provider import.
typedef struct loom_bytecode_reader_provider_import_t {
  // Validated STRINGS provider key.
  loom_string_id_t provider_id;
  // First entry in provider_import_anchors.
  uint32_t first_anchor_index;
  // Number of anchors in the contiguous slice.
  uint32_t anchor_count;
  // Validated source comments and authored vertical separation.
  loom_bytecode_source_trivia_t source_trivia;
} loom_bytecode_reader_provider_import_t;

// Immutable facts established while validating one module. All pointers
// borrow either input bytecode or storage from the reader arenas.
typedef struct loom_bytecode_reader_module_view_t {
  // Validated file-directory entry for this module.
  const loom_bytecode_reader_module_t* directory_entry;
  // Optional caller-owned retained metadata projection target.
  struct loom_bytecode_module_metadata_t* output_metadata;
  // Validated module allocation and table-count summary.
  loom_bytecode_module_metadata_summary_t summary;
  // Validated file header retained for output module construction.
  loom_bytecode_source_trivia_t file_header;

  // Validated section directory and direct section identities.
  struct {
    // Dense section directory in wire order.
    loom_bytecode_reader_section_t* values;
    // Number of entries in the section directory.
    iree_host_size_t count;
    // Required STRINGS section.
    const loom_bytecode_reader_section_t* strings;
    // Required SOURCES section.
    const loom_bytecode_reader_section_t* sources;
    // Required ENCODINGS section.
    const loom_bytecode_reader_section_t* encodings;
    // Required TYPES section.
    const loom_bytecode_reader_section_t* types;
    // Required OPS section.
    const loom_bytecode_reader_section_t* ops;
    // Optional LOCATIONS section.
    const loom_bytecode_reader_section_t* locations;
    // Optional SOURCE_TRIVIA section.
    const loom_bytecode_reader_section_t* source_trivia;
    // Required SYMBOLS section.
    const loom_bytecode_reader_section_t* symbols;
    // Required PROVIDER_IMPORTS section.
    const loom_bytecode_reader_section_t* provider_imports;
    // Required SYMBOL_REFERENCES section.
    const loom_bytecode_reader_section_t* symbol_references;
    // Required IR section.
    const loom_bytecode_reader_section_t* ir;
  } sections;

  // Validated module string table.
  struct {
    // Borrowed UTF-8 string views in wire order.
    iree_string_view_t* values;
    // Number of strings.
    iree_host_size_t count;
  } strings;

  // Validated source-name table.
  struct {
    // Borrowed UTF-8 source-name views in wire order.
    iree_string_view_t* values;
    // Number of sources.
    iree_host_size_t count;
  } sources;

  // Immutable type materialization plan.
  struct {
    // Dense direct entries and diagnostic offsets in wire order.
    loom_bytecode_type_plan_entry_t* entries;
    // Number of types.
    iree_host_size_t count;
    // Sparse non-direct facts in wire order, or NULL when all types are direct.
    loom_bytecode_type_fact_t* facts;
  } types;

  // Validated registered operation table.
  struct {
    // Resolved operation vtables in wire order.
    const loom_op_vtable_t** values;
    // Resolved operation kinds in wire order.
    loom_op_kind_t* kinds;
    // Number of registered operations.
    iree_host_size_t count;
  } ops;

  // Validated encoding table facts.
  struct {
    // Encoding-family name string IDs in wire order.
    loom_string_id_t* family_name_ids;
    // Number of encoding families.
    iree_host_size_t family_count;
    // Number of encoding instances.
    iree_host_size_t count;
  } encodings;

  // Validated location table facts.
  struct {
    // Number of locations.
    iree_host_size_t count;
  } locations;

  // Validated symbol table identities.
  struct {
    // Number of symbols.
    iree_host_size_t count;
    // Symbol ordinal to validated STRINGS name ID.
    loom_string_id_t* name_ids;
    // Symbol ordinal to defining operation-table ordinal, or UINT32_MAX.
    uint32_t* defining_op_ordinals;
    // Symbol ordinal to validated wire flags.
    loom_bytecode_symbol_flags_t* flags;
    // Symbol ordinal to validated wire kind.
    uint8_t* kinds;
    // Number of unresolved wire-only provider anchors.
    iree_host_size_t unresolved_anchor_count;
    // Name ID to symbol ordinal index.
    loom_symbol_map_t map;
  } symbols;

  // Validated compile-time provider imports.
  struct {
    // Canonically ordered provider records.
    loom_bytecode_reader_provider_import_t* values;
    // Number of provider records.
    iree_host_size_t count;
    // Flat module-local symbol references sliced by provider records.
    loom_symbol_ref_t* anchors;
    // Number of provider anchors.
    iree_host_size_t anchor_count;
  } provider_imports;
} loom_bytecode_reader_module_view_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_MODULE_VIEW_H_
