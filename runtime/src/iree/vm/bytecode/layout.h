// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_LAYOUT_H_
#define IREE_VM_BYTECODE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/module.h"
#include "iree/vm/function.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable mapped views produced by structural verification. All pointers
// borrow the exact image passed to verification and remain valid only while
// those bytes remain live and bitwise stable.

// Mapped canonical image envelope.
typedef struct iree_vm_bytecode_image_layout_t {
  // Fixed image header at byte zero.
  const iree_vm_bytecode_v0_image_header_t* header;
  // Strictly type-sorted section directory rows.
  const iree_vm_bytecode_v0_section_directory_row_t* sections;
  // Number of rows in |sections|.
  uint16_t section_count;
} iree_vm_bytecode_image_layout_t;

// Mapped architectural extension requirements.
typedef struct iree_vm_bytecode_requirement_table_t {
  // Strictly page-sorted requirement rows.
  const iree_vm_bytecode_v0_requirement_row_t* rows;
  // Number of rows in |rows|.
  uint16_t count;
} iree_vm_bytecode_requirement_table_t;

// Mapped canonical string table.
typedef struct iree_vm_bytecode_string_table_t {
  // Count-plus-one offsets into |data|.
  const iree_vm_bytecode_v0_string_offset_t* offsets;
  // Exact UTF-8 byte tail.
  const uint8_t* data;
  // Number of indexed strings.
  uint32_t count;
  // Number of bytes in |data|.
  uint32_t data_length;
} iree_vm_bytecode_string_table_t;

// Mapped canonical ref-type declarations.
typedef struct iree_vm_bytecode_ref_type_table_t {
  // Namespace group rows.
  const iree_vm_bytecode_v0_ref_type_group_row_t* groups;
  // Flat type entry rows.
  const iree_vm_bytecode_v0_ref_type_entry_row_t* entries;
  // Number of rows in |groups|.
  uint32_t group_count;
  // Number of rows in |entries|.
  uint32_t entry_count;
} iree_vm_bytecode_ref_type_table_t;

// Mapped canonical machine signatures.
typedef struct iree_vm_bytecode_signature_table_t {
  // Source-ordered signature rows.
  const iree_vm_bytecode_v0_signature_row_t* rows;
  // Source-ordered descriptor rows.
  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors;
  // Number of rows in |rows|.
  uint32_t count;
  // Number of rows in |descriptors|.
  uint32_t descriptor_count;
} iree_vm_bytecode_signature_table_t;

// Mapped canonical callable types.
typedef struct iree_vm_bytecode_callable_type_table_t {
  // Canonically ordered callable type rows.
  const iree_vm_bytecode_v0_callable_type_row_t* rows;
  // Number of rows in |rows|.
  uint32_t count;
} iree_vm_bytecode_callable_type_table_t;

// Mapped canonical import declarations.
typedef struct iree_vm_bytecode_import_table_t {
  // Target-module group rows.
  const iree_vm_bytecode_v0_import_group_row_t* groups;
  // Flat import entry rows.
  const iree_vm_bytecode_v0_import_entry_row_t* entries;
  // Number of rows in |groups|.
  uint32_t group_count;
  // Number of rows in |entries|.
  uint32_t entry_count;
} iree_vm_bytecode_import_table_t;

// Mapped canonical export declarations.
typedef struct iree_vm_bytecode_export_table_t {
  // Name-sorted export rows.
  const iree_vm_bytecode_v0_export_row_t* rows;
  // Number of rows in |rows|.
  uint32_t count;
} iree_vm_bytecode_export_table_t;

// Mapped canonical function definitions.
typedef struct iree_vm_bytecode_function_table_t {
  // Function declaration rows.
  const iree_vm_bytecode_v0_function_row_t* rows;
  // Function-ordered switch-target entries.
  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets;
  // Concatenated instruction record streams.
  const uint8_t* bytecode_data;
  // Number of rows in |rows|.
  uint32_t count;
  // Number of rows in |switch_targets|.
  uint32_t switch_target_count;
  // Exact byte length of |bytecode_data|.
  uint32_t bytecode_length;
} iree_vm_bytecode_function_table_t;

// Mapped canonical constant pool.
typedef struct iree_vm_bytecode_constant_table_t {
  // Naturally aligned constant cells.
  const iree_vm_bytecode_v0_constant_cell_t* cells;
  // Number of rows in |cells|.
  uint32_t count;
} iree_vm_bytecode_constant_table_t;

// Mapped canonical global declarations.
typedef struct iree_vm_bytecode_global_table_t {
  // Three-domain global count header, or null when globals are absent.
  const iree_vm_bytecode_v0_globals_header_t* header;
  // Ref-global descriptor rows.
  const iree_vm_bytecode_v0_global_ref_descriptor_row_t* refs;
  // Function-global descriptor rows.
  const iree_vm_bytecode_v0_global_function_descriptor_row_t* functions;
} iree_vm_bytecode_global_table_t;

// Mapped canonical rodata storage.
typedef struct iree_vm_bytecode_rodata_table_t {
  // Block descriptors in ordinal order.
  const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptors;
  // First byte of the complete rodata section.
  const uint8_t* section_begin;
  // Number of rows in |descriptors|.
  uint32_t count;
  // Section-relative byte offset after the descriptor array.
  iree_host_size_t blocks_offset;
} iree_vm_bytecode_rodata_table_t;

// Mapped canonical authored presentation.
typedef struct iree_vm_bytecode_presentation_table_t {
  // Declaration-sorted presentation rows.
  const iree_vm_bytecode_v0_presentation_entry_row_t* entries;
  // Argument-then-result field rows.
  const iree_vm_bytecode_v0_presentation_field_row_t* fields;
  // Number of rows in |entries|.
  uint32_t entry_count;
  // Number of rows in |fields|.
  uint32_t field_count;
} iree_vm_bytecode_presentation_table_t;

// Mapped canonical typed metadata.
typedef struct iree_vm_bytecode_metadata_table_t {
  // Metadata count header, or null when metadata is absent.
  const iree_vm_bytecode_v0_metadata_header_t* header;
  // Sparse nonempty import scopes.
  const iree_vm_bytecode_v0_metadata_scope_row_t* import_scopes;
  // Sparse nonempty export scopes.
  const iree_vm_bytecode_v0_metadata_scope_row_t* export_scopes;
  // Complete scope-partitioned metadata entries.
  const iree_vm_bytecode_v0_metadata_entry_row_t* entries;
  // Count-plus-one offsets into |value_data|.
  const iree_vm_bytecode_v0_metadata_value_offset_t* value_offsets;
  // Exact concatenated metadata value bytes.
  const uint8_t* value_data;
  // Number of bytes in |value_data|.
  iree_host_size_t value_data_length;
} iree_vm_bytecode_metadata_table_t;

// Complete mapped view of one structurally verified module image.
typedef struct iree_vm_bytecode_module_layout_t {
  // Canonical image envelope.
  iree_vm_bytecode_image_layout_t image;
  // Architectural extension requirements.
  iree_vm_bytecode_requirement_table_t requirements;
  // Canonical string table.
  iree_vm_bytecode_string_table_t strings;
  // Canonical ref-type table.
  iree_vm_bytecode_ref_type_table_t ref_types;
  // Canonical machine signature table.
  iree_vm_bytecode_signature_table_t signatures;
  // Canonical callable-type table.
  iree_vm_bytecode_callable_type_table_t callable_types;
  // Canonical import table.
  iree_vm_bytecode_import_table_t imports;
  // Canonical export table.
  iree_vm_bytecode_export_table_t exports;
  // Canonical function table.
  iree_vm_bytecode_function_table_t functions;
  // Canonical constant pool.
  iree_vm_bytecode_constant_table_t constants;
  // Canonical global table.
  iree_vm_bytecode_global_table_t globals;
  // Canonical rodata table.
  iree_vm_bytecode_rodata_table_t rodata;
  // Optional authored presentation table.
  iree_vm_bytecode_presentation_table_t presentation;
  // Optional typed metadata table.
  iree_vm_bytecode_metadata_table_t metadata;
} iree_vm_bytecode_module_layout_t;

// Private construction state at the start of every nonempty process slice.
typedef struct iree_vm_bytecode_process_header_t {
  // Open or sealed construction state.
  uint32_t construction_state;
  // Reserved zero word preserving natural cell alignment.
  uint32_t reserved;
} iree_vm_bytecode_process_header_t;

// Private byte offsets assigned within one process storage slice.
typedef struct iree_vm_bytecode_process_layout_t {
  // Exact opaque process storage byte length.
  iree_host_size_t total_size;
  // First value-global cell.
  iree_host_size_t values_offset;
  // First ref-global cell.
  iree_host_size_t refs_offset;
  // First function-global cell.
  iree_host_size_t functions_offset;
  // First immutable value-global set-bit word.
  iree_host_size_t value_set_bits_offset;
  // First immutable ref-global set-bit word.
  iree_host_size_t ref_set_bits_offset;
  // First immutable function-global set-bit word.
  iree_host_size_t function_set_bits_offset;
} iree_vm_bytecode_process_layout_t;

// Exact storage needed for rodata blocks that cannot map the image directly.
typedef struct iree_vm_bytecode_rodata_storage_plan_t {
  // Total byte length of packed fallback copies.
  iree_host_size_t copy_length;
  // Maximum alignment required by a fallback copy, or zero when empty.
  iree_host_size_t copy_alignment;
} iree_vm_bytecode_rodata_storage_plan_t;

// Bounded allocation facts derived before constructing persistent storage.
typedef struct iree_vm_bytecode_module_plan_t {
  // Complete mapped image layout.
  iree_vm_bytecode_module_layout_t layout;
  // Largest decoded control.block count among all functions.
  uint32_t maximum_block_count;
  // Exact process-storage layout.
  iree_vm_bytecode_process_layout_t process_layout;
  // Exact fallback storage for image-misaligned rodata.
  iree_vm_bytecode_rodata_storage_plan_t rodata_storage;
} iree_vm_bytecode_module_plan_t;

// Returns one valid string ordinal as a stable image-backed view.
static inline iree_string_view_t iree_vm_bytecode_string_at(
    const iree_vm_bytecode_string_table_t* table, uint16_t ordinal) {
  const uint32_t begin = table->offsets[ordinal].byte_offset_u32;
  const uint32_t end = table->offsets[ordinal + 1].byte_offset_u32;
  return iree_make_string_view((const char*)table->data + begin, end - begin);
}

// Returns a nullable string ordinal as an empty or stable image-backed view.
static inline iree_string_view_t iree_vm_bytecode_nullable_string_at(
    const iree_vm_bytecode_string_table_t* table, uint16_t ordinal) {
  return ordinal == UINT16_MAX ? iree_string_view_empty()
                               : iree_vm_bytecode_string_at(table, ordinal);
}

// Returns the source-ordered descriptors for one valid signature ordinal.
static inline const iree_vm_bytecode_v0_signature_descriptor_row_t*
iree_vm_bytecode_signature_descriptors(
    const iree_vm_bytecode_signature_table_t* table, uint16_t ordinal) {
  return table->descriptors + table->rows[ordinal].descriptor_base_u32;
}

// Returns the callable type implemented by one verified function.
static inline const iree_vm_bytecode_v0_callable_type_row_t*
iree_vm_bytecode_function_callable_type(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function) {
  return &layout->callable_types.rows[function->callable_type_ordinal_u16];
}

// Returns the signature implemented by one verified function.
static inline const iree_vm_bytecode_v0_signature_row_t*
iree_vm_bytecode_function_signature(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function) {
  return &layout->signatures
              .rows[iree_vm_bytecode_function_callable_type(layout, function)
                        ->signature_ordinal_u16];
}

// Returns the source-ordered argument count of one verified signature.
static inline uint32_t iree_vm_bytecode_signature_argument_count(
    const iree_vm_bytecode_v0_signature_row_t* row) {
  return (uint32_t)row->argument_value_count_u16 +
         (uint32_t)row->argument_ref_count_u16 +
         (uint32_t)row->argument_function_count_u16;
}

// Returns the source-ordered result count of one verified signature.
static inline uint32_t iree_vm_bytecode_signature_result_count(
    const iree_vm_bytecode_v0_signature_row_t* row) {
  return (uint32_t)row->result_value_count_u16 +
         (uint32_t)row->result_ref_count_u16 +
         (uint32_t)row->result_function_count_u16;
}

// Returns the exact instruction bytes of one verified function.
static inline iree_const_byte_span_t iree_vm_bytecode_function_data(
    const iree_vm_bytecode_function_table_t* table,
    const iree_vm_bytecode_v0_function_row_t* function) {
  return iree_make_const_byte_span(
      table->bytecode_data + function->bytecode_offset_u32,
      function->bytecode_length_u32);
}

// Returns the exact switch-target rows of one verified function.
static inline const iree_vm_bytecode_v0_switch_target_entry_t*
iree_vm_bytecode_function_switch_targets(
    const iree_vm_bytecode_function_table_t* table,
    const iree_vm_bytecode_v0_function_row_t* function) {
  return table->switch_targets + function->switch_target_base_u32;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_LAYOUT_H_
