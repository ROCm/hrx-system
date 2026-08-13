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
#include "loom/ir/attribute.h"
#include "loom/ir/symbol_map.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"

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
  // Authored leading vertical separation.
  bool leading_blank_line;
  // Borrowed comment payload views.
  const iree_string_view_t* comments;
  // Number of attached comments.
  uint16_t comment_count;
} loom_bytecode_reader_provider_import_t;

// One dense validated type-table entry.
typedef struct loom_bytecode_type_plan_entry_t {
  // Complete by-value type when the entry has no sparse fact.
  loom_type_t direct_type;
  // Absolute bytecode offset of the entry kind.
  uint64_t bytecode_offset;
} loom_bytecode_type_plan_entry_t;

// Common header for one sparse fact in topological type-table order.
typedef struct loom_bytecode_type_fact_t {
  // Next sparse fact in increasing type ID order.
  struct loom_bytecode_type_fact_t* next;
  // Dense type-table entry described by this fact.
  loom_type_id_t type_id;
  // Non-direct loom_type_kind_t discriminator.
  loom_type_kind_t kind;
} loom_bytecode_type_fact_t;

// Type-reference facts for one function type.
typedef struct loom_bytecode_function_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // Number of leading argument type IDs.
  uint16_t argument_count;
  // Number of trailing result type IDs.
  uint16_t result_count;
  // Argument then result type IDs in wire order.
  loom_type_id_t type_ids[];
} loom_bytecode_function_type_fact_t;

// Type-reference facts for one dialect type.
typedef struct loom_bytecode_dialect_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // Validated STRINGS family name ID.
  loom_string_id_t name_id;
  // Number of parameter type IDs.
  uint16_t parameter_count;
  // Parameter type IDs in wire order.
  loom_type_id_t type_ids[];
} loom_bytecode_dialect_type_fact_t;

// Materialization facts for one descriptor-backed type.
typedef struct loom_bytecode_parameterized_type_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // Resolved static family descriptor.
  const loom_parameterized_type_descriptor_t* descriptor;
  // Absolute offset of the first present parameter name.
  uint64_t parameters_offset;
  // Exact byte length of all present parameter names, kinds, and values.
  iree_host_size_t parameters_length;
  // Number of present parameters in the exact payload.
  uint8_t present_count;
  // Validated descriptor indices for present parameters in wire order.
  uint8_t parameter_indices[];
} loom_bytecode_parameterized_type_fact_t;

// Materialization facts for one typed target-register payload.
typedef struct loom_bytecode_typed_register_fact_t {
  // Common sparse type-fact header.
  loom_bytecode_type_fact_t base;
  // First target-owned carrier payload word.
  uint64_t carrier_payload0;
  // Second target-owned carrier payload word.
  uint64_t carrier_payload1;
  // Prior semantic value type ID.
  loom_type_id_t value_type_id;
} loom_bytecode_typed_register_fact_t;

// Immutable facts established while validating one module. All pointers
// borrow either input bytecode or storage from the reader arenas.
typedef struct loom_bytecode_reader_module_view_t {
  // Validated file-directory entry for this module.
  const loom_bytecode_reader_module_t* directory_entry;
  // Optional caller-owned retained metadata projection target.
  struct loom_bytecode_module_metadata_t* output_metadata;
  // Validated module allocation and table-count summary.
  loom_bytecode_module_metadata_summary_t summary;

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
