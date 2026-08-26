// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bytecode metadata validator and retained index.
//
// This is the metadata-only front door for untrusted .loombc input. Validation
// resolves file/module/section structure and lightweight tables without
// materializing IR bodies. Retained indexes additionally preserve borrowed
// names, reference rows, and exact entry/body ranges for later materialization.

#ifndef LOOM_FORMAT_BYTECODE_INDEX_H_
#define LOOM_FORMAT_BYTECODE_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/module_summary.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling metadata validation and retained index construction.
typedef struct loom_bytecode_index_options_t {
  // Sink for structured malformed-bytecode diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
} loom_bytecode_index_options_t;

// Validated section directory entry exposed by the bytecode index.
typedef struct loom_bytecode_section_metadata_t {
  // Wire section kind.
  uint16_t kind;
  // Validated section flags.
  loom_bytecode_section_flags_t flags;
  // Module-relative byte offset of the section payload.
  uint64_t offset;
  // Byte length of the section payload.
  uint64_t length;
  // Absolute file byte offset of the section payload.
  uint64_t absolute_offset;
  // Borrowed byte span for the section payload.
  iree_const_byte_span_t bytes;
} loom_bytecode_section_metadata_t;

// Validated byte range for one variable-length shared-table entry.
typedef struct loom_bytecode_table_entry_metadata_t {
  // Absolute file byte offset of the entry.
  uint64_t entry_offset;
  // Byte length of the entry.
  uint64_t entry_length;
} loom_bytecode_table_entry_metadata_t;

// Validated ENCODINGS instance record exposed by the bytecode index.
typedef struct loom_bytecode_encoding_metadata_t {
  // Absolute file byte offset of the instance entry.
  uint64_t entry_offset;
  // Byte length of the instance entry.
  uint64_t entry_length;
  // Source STRINGS ordinal naming the encoding family.
  uint32_t name_string_index;
} loom_bytecode_encoding_metadata_t;

// Validated OPS record exposed by the bytecode index.
typedef struct loom_bytecode_op_metadata_t {
  // Borrowed registered operation name view from the module STRINGS table.
  iree_string_view_t name;
  // Context-owned operation vtable resolved during bytecode validation.
  const loom_op_vtable_t* vtable;
  // Dense context-local operation kind resolved during bytecode validation.
  loom_op_kind_t kind;
} loom_bytecode_op_metadata_t;

// Validated independently bounded root-region payload.
typedef struct loom_bytecode_region_payload_metadata_t {
  // IR-section-relative payload byte offset.
  uint64_t offset;
  // Absolute file byte offset of the payload.
  uint64_t absolute_offset;
  // Byte length of the payload.
  uint32_t length;
  // Declared region slot on the defining symbol operation.
  uint8_t region_index;
} loom_bytecode_region_payload_metadata_t;

// Validated function/global/record attribute payload in one symbol entry.
typedef struct loom_bytecode_symbol_attribute_metadata_t {
  // Absolute file byte offset of the attribute value payload after its kind.
  uint64_t value_offset;
  // Byte length of the attribute value payload.
  uint64_t value_length;
  // Descriptor slot on the defining operation.
  uint8_t attribute_index;
  // Validated wire attribute kind.
  loom_bytecode_attr_kind_t kind;
} loom_bytecode_symbol_attribute_metadata_t;

// Validated symbol record exposed by the bytecode index.
typedef struct loom_bytecode_symbol_metadata_t {
  // Absolute file byte offset of the symbol entry.
  uint64_t entry_offset;
  // Byte length of the symbol entry.
  uint64_t entry_length;
  // Borrowed symbol name view from the module STRINGS table.
  iree_string_view_t name;
  // Source STRINGS ordinal naming the symbol.
  uint32_t name_string_index;
  // Wire symbol kind.
  loom_bytecode_symbol_kind_t kind;
  // Wire symbol visibility.
  loom_bytecode_symbol_visibility_t visibility;
  // Validated symbol flags.
  loom_bytecode_symbol_flags_t flags;
  // Structural symbol interfaces declared by the defining op metadata.
  uint32_t interfaces;
  // Borrowed source module name for imports, or empty when not imported.
  iree_string_view_t import_module;
  // Borrowed source symbol name for imports, or empty when not imported.
  iree_string_view_t import_symbol;
  // Module-local OPS ordinal of the defining operation, or UINT32_MAX for
  // symbols without a defining operation.
  uint32_t defining_op_ordinal;
  // Function-like calling convention byte, or zero for non-function symbols.
  uint8_t calling_convention;
  // Function-like purity byte, or zero for non-function symbols.
  uint8_t purity;
  // Kernel workload argument count, or zero for ordinary functions.
  uint16_t kernel_workload_argument_count;
  // Function-like argument count, or zero for non-function symbols.
  uint16_t argument_count;
  // Function/global result count, or zero when the symbol has no signature.
  uint16_t result_count;
  // Function-like tied result count.
  uint16_t tied_result_count;
  // FuncLike body region index plus one, or zero when absent.
  uint8_t body_region_index_plus_one;
  // Body entry in this symbol's payload list plus one, or zero when absent.
  uint8_t body_region_payload_ordinal_plus_one;
  // Kernel workload/configuration region index plus one, or zero when absent.
  uint8_t kernel_workload_region_index_plus_one;
  // Workload/configuration entry in this symbol's payload list plus one, or
  // zero when absent.
  uint8_t kernel_workload_region_payload_ordinal_plus_one;
  // Number of root region slots declared by the defining operation.
  uint8_t root_region_count;
  // Global declaration-local value count, or zero for non-global symbols.
  uint64_t local_value_count;
  // Source SYMBOLS ordinal naming the provider's template family, or
  // UINT32_MAX for non-provider symbols.
  uint32_t template_family_symbol_ordinal;
  // Concrete provider priority value, or zero when not applicable.
  uint64_t priority;
  // Arena-owned predicates retaining signature-local VALUE ordinals.
  const loom_predicate_t* predicates;
  // Number of retained predicate entries.
  uint16_t predicate_count;
  // Arena-owned attribute payloads required by retained linker contracts.
  const loom_bytecode_symbol_attribute_metadata_t* attributes;
  // Number of retained attribute payloads.
  uint8_t attribute_count;
  // Retained template target attribute ordinal plus one, or zero when absent.
  uint8_t template_target_attribute_ordinal_plus_one;
  // Retained template requires attribute ordinal plus one, or zero when absent.
  uint8_t template_requires_attribute_ordinal_plus_one;
  // First entry in the module's root-region payload array.
  uint32_t first_region_payload_index;
  // Number of independently bounded root-region payloads.
  uint8_t region_payload_count;
} loom_bytecode_symbol_metadata_t;

// One compile-time provider import and its slice in the module anchor array.
typedef struct loom_bytecode_provider_import_metadata_t {
  // Borrowed provider key view from the module STRINGS table.
  iree_string_view_t provider;
  // First entry in provider_import_anchor_symbol_indices.
  uint32_t first_anchor_index;
  // Number of symbol ordinals in this provider's contiguous slice.
  uint32_t anchor_count;
  // True when the import has authored leading vertical separation.
  bool leading_blank_line;
  // Arena-owned array of views borrowing authored comment payloads.
  const iree_string_view_t* comments;
  // Number of authored comments.
  uint16_t comment_count;
} loom_bytecode_provider_import_metadata_t;

// Dependency and abstract-provider slices owned by one SYMBOLS ordinal.
typedef struct loom_bytecode_symbol_reference_metadata_t {
  // First entry in dependency_symbol_indices.
  uint32_t first_dependency_index;
  // Number of dependency occurrences in this symbol's slice.
  uint32_t dependency_count;
  // First entry in template_demands.
  uint32_t first_template_demand_index;
  // Number of abstract provider demands in this symbol's slice.
  uint32_t template_demand_count;
} loom_bytecode_symbol_reference_metadata_t;

// Validated module directory entry and lightweight per-module index.
typedef struct loom_bytecode_module_metadata_t {
  // Borrowed module name view from the file string pool.
  iree_string_view_t name;
  // Validated module directory flags.
  loom_bytecode_module_flags_t flags;
  // Absolute file byte offset of the module payload.
  uint64_t offset;
  // Byte length of the module payload.
  uint64_t length;
  // Allocation and table-count summary decoded from the module.
  loom_bytecode_module_metadata_summary_t summary;
  // Number of validated section directory entries.
  iree_host_size_t section_count;
  // Arena-owned section metadata array.
  loom_bytecode_section_metadata_t* sections;
  // Validated module-local string table.
  struct {
    // Number of borrowed string views.
    iree_host_size_t count;
    // Arena-owned array of views borrowing from the input bytecode.
    iree_string_view_t* values;
  } strings;
  // Validated module-local source-name table.
  struct {
    // Number of borrowed source-name views.
    iree_host_size_t count;
    // Arena-owned array of views borrowing from the input bytecode.
    iree_string_view_t* values;
  } sources;
  // Seekable module-local type table.
  struct {
    // Number of validated type entries.
    iree_host_size_t count;
    // Arena-owned entry range array.
    loom_bytecode_table_entry_metadata_t* entries;
  } types;
  // Seekable module-local encoding instance table.
  struct {
    // Number of validated encoding instances.
    iree_host_size_t count;
    // Arena-owned encoding metadata array.
    loom_bytecode_encoding_metadata_t* entries;
  } encodings;
  // Resolved module-local operation table.
  struct {
    // Number of validated operation records.
    iree_host_size_t count;
    // Arena-owned operation metadata array.
    loom_bytecode_op_metadata_t* entries;
  } ops;
  // Seekable module-local location table.
  struct {
    // Number of validated location entries.
    iree_host_size_t count;
    // Arena-owned entry range array.
    loom_bytecode_table_entry_metadata_t* entries;
  } locations;
  // Number of validated symbol records.
  iree_host_size_t symbol_count;
  // Arena-owned symbol metadata array.
  loom_bytecode_symbol_metadata_t* symbols;
  // Number of independently bounded root-region payloads.
  iree_host_size_t region_payload_count;
  // Arena-owned payload metadata in symbol and declared region-slot order.
  loom_bytecode_region_payload_metadata_t* region_payloads;
  // Dense STRINGS ordinal to SYMBOLS ordinal projection. Entries without a
  // symbol contain UINT32_MAX.
  uint32_t* symbol_ordinal_by_string_index;
  // Number of import offset table entries.
  iree_host_size_t import_count;
  // Arena-owned symbol indices in import offset table order.
  uint32_t* import_symbol_indices;
  // Number of export offset table entries.
  iree_host_size_t export_count;
  // Arena-owned symbol indices in export offset table order.
  uint32_t* export_symbol_indices;
  // Number of compile-time provider import records.
  iree_host_size_t provider_import_count;
  // Arena-owned provider records in canonical provider order.
  loom_bytecode_provider_import_metadata_t* provider_imports;
  // Number of entries in the flat provider anchor ordinal array.
  iree_host_size_t provider_import_anchor_count;
  // Arena-owned SYMBOLS ordinals sliced by provider_imports.
  uint32_t* provider_import_anchor_symbol_indices;
  // Number of module-root dependency ordinals at the start of
  // dependency_symbol_indices.
  uint32_t module_dependency_count;
  // Total number of dependency occurrences across all rows.
  iree_host_size_t dependency_count;
  // Arena-owned dependency target ordinals in deterministic occurrence order.
  // The module-root slice begins at zero; symbol slices are described by
  // symbol_references. Targets may repeat within a slice.
  uint32_t* dependency_symbol_indices;
  // Arena-owned target interface constraints parallel to
  // dependency_symbol_indices. Zero accepts any target interface.
  loom_symbol_interface_flags_t* dependency_target_interfaces;
  // Arena-owned source root region indices plus one parallel to
  // dependency_symbol_indices. Zero identifies a source symbol contract or
  // module-root occurrence.
  uint8_t* dependency_source_root_region_indices_plus_one;
  // One dependency and template-demand slice per SYMBOLS ordinal.
  loom_bytecode_symbol_reference_metadata_t* symbol_references;
  // Total number of template-family demand occurrences.
  iree_host_size_t template_demand_count;
  // Arena-owned family SYMBOLS ordinals in deterministic demand order and
  // sliced by symbol_references. Ordinals may repeat within a slice.
  uint32_t* template_demand_family_symbol_ordinals;
  // Arena-owned source root region indices plus one parallel to
  // template_demand_family_symbol_ordinals.
  uint8_t* template_demand_source_root_region_indices_plus_one;
} loom_bytecode_module_metadata_t;

// Resolves a source STRINGS ordinal to its module-local SYMBOLS ordinal.
static inline bool loom_bytecode_module_metadata_lookup_symbol_ordinal(
    const loom_bytecode_module_metadata_t* metadata,
    uint32_t source_name_string_ordinal, uint32_t* out_source_symbol_ordinal) {
  if (source_name_string_ordinal >= metadata->strings.count ||
      metadata->symbol_ordinal_by_string_index == NULL) {
    return false;
  }
  const uint32_t symbol_ordinal =
      metadata->symbol_ordinal_by_string_index[source_name_string_ordinal];
  if (symbol_ordinal == UINT32_MAX) {
    return false;
  }
  *out_source_symbol_ordinal = symbol_ordinal;
  return true;
}

// Validated file-level bytecode index.
//
// Arrays in this structure and its child records are allocated from the
// caller-owned arena passed to loom_bytecode_read_index. String views and byte
// spans borrow from the input bytecode buffer. Callers must keep both the arena
// and bytecode bytes alive while using the index.
typedef struct loom_bytecode_file_metadata_t {
  // Bytecode format version from the file header.
  uint8_t format_version;
  // File-level source-location mode from the file header.
  loom_bytecode_location_mode_t location_mode;
  // Producer string from the file header.
  iree_string_view_t producer;
  // Number of decoded module directory entries.
  iree_host_size_t module_count;
  // Arena-owned module metadata array.
  loom_bytecode_module_metadata_t* modules;
} loom_bytecode_file_metadata_t;

// Result of reading bytecode metadata.
typedef struct loom_bytecode_read_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of warning diagnostics emitted.
  uint32_t warning_count;
  // Number of module directory entries decoded.
  uint16_t module_count;
  // File-level source-location mode from the header.
  loom_bytecode_location_mode_t location_mode;
  // Summary for the first module. Multi-module validation is supported, but
  // this V1 result keeps only the first summary until archive-level APIs need
  // per-module output.
  loom_bytecode_module_metadata_summary_t first_module;
} loom_bytecode_read_result_t;

// Validates .loombc file structure and metadata sections.
//
// Malformed bytecode is reported through |options->diagnostic_sink| and in
// |out_result->error_count|. The function returns a non-OK status only for
// infrastructure failures such as OOM or a sink callback failure.
//
// |context| must be finalized and contain the dialect/encoding registrations
// used to resolve OPS and ENCODINGS metadata. |block_pool| provides transient
// arena storage for decoded metadata and is reset before return.
iree_status_t loom_bytecode_read_metadata(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_index_options_t* options,
    loom_bytecode_read_result_t* out_result);

// Reads and validates the bytecode file index without materializing IR bodies.
//
// The metadata arena owns all arrays stored in |out_metadata|. It is not reset
// by this function. |block_pool| provides transient scratch storage that is
// returned before this function returns.
iree_status_t loom_bytecode_read_index(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* metadata_arena,
    const loom_bytecode_index_options_t* options,
    loom_bytecode_read_result_t* out_result,
    loom_bytecode_file_metadata_t* out_metadata);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_INDEX_H_
