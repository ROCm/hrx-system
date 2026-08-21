// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_STORAGE_H_
#define IREE_VM_BYTECODE_MODULE_STORAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/vm/buffer_provider.h"
#include "iree/vm/bytecode/storage.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "iree/vm/module.h"

// Mapped canonical image envelope.
typedef struct iree_vm_bytecode_image_layout_t {
  // Fixed image header at byte zero.
  const iree_vm_bytecode_v0_image_header_t* header;
  // Number of section directory rows.
  uint16_t section_count;
  // Strictly type-sorted section directory rows.
  const iree_vm_bytecode_v0_section_directory_row_t* sections;
} iree_vm_bytecode_image_layout_t;

// Mapped architectural extension requirements.
typedef struct iree_vm_bytecode_requirement_table_t {
  // Number of declared extension pages.
  uint16_t count;
  // Strictly page-sorted requirement rows.
  const iree_vm_bytecode_v0_requirement_row_t* rows;
} iree_vm_bytecode_requirement_table_t;

// Mapped canonical string table.
typedef struct iree_vm_bytecode_string_table_t {
  // Number of indexed strings.
  uint32_t count;
  // Count-plus-one offsets into |data|.
  const iree_vm_bytecode_v0_string_offset_t* offsets;
  // Exact UTF-8 byte tail.
  const uint8_t* data;
  // Number of bytes in |data|.
  uint32_t data_length;
} iree_vm_bytecode_string_table_t;

// Mapped canonical ref-type declarations.
typedef struct iree_vm_bytecode_ref_type_table_t {
  // Number of namespace groups.
  uint32_t group_count;
  // Namespace group rows.
  const iree_vm_bytecode_v0_ref_type_group_row_t* groups;
  // Flat type entry count.
  uint32_t entry_count;
  // Flat type entry rows.
  const iree_vm_bytecode_v0_ref_type_entry_row_t* entries;
} iree_vm_bytecode_ref_type_table_t;

// Mapped canonical machine signatures.
typedef struct iree_vm_bytecode_signature_table_t {
  // Number of source-ordered signatures.
  uint32_t count;
  // Signature rows.
  const iree_vm_bytecode_v0_signature_row_t* rows;
  // Number of logical descriptors.
  uint32_t descriptor_count;
  // Source-ordered descriptor rows.
  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors;
} iree_vm_bytecode_signature_table_t;

// Mapped canonical callable types.
typedef struct iree_vm_bytecode_callable_type_table_t {
  // Number of callable types.
  uint32_t count;
  // Callable type rows.
  const iree_vm_bytecode_v0_callable_type_row_t* rows;
} iree_vm_bytecode_callable_type_table_t;

// Mapped canonical import declarations.
typedef struct iree_vm_bytecode_import_table_t {
  // Number of target-module groups.
  uint32_t group_count;
  // Target-module group rows.
  const iree_vm_bytecode_v0_import_group_row_t* groups;
  // Flat import entry count.
  uint32_t entry_count;
  // Flat import entry rows.
  const iree_vm_bytecode_v0_import_entry_row_t* entries;
} iree_vm_bytecode_import_table_t;

// Mapped canonical export declarations.
typedef struct iree_vm_bytecode_export_table_t {
  // Number of export rows.
  uint32_t count;
  // Name-sorted export rows.
  const iree_vm_bytecode_v0_export_row_t* rows;
} iree_vm_bytecode_export_table_t;

// Mapped canonical function definitions.
typedef struct iree_vm_bytecode_function_table_t {
  // Number of function rows.
  uint32_t count;
  // Function rows.
  const iree_vm_bytecode_v0_function_row_t* rows;
  // Number of switch-target entries.
  uint32_t switch_target_count;
  // Function-ordered switch-target entries.
  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets;
  // Concatenated instruction record streams.
  const uint8_t* bytecode_data;
  // Exact byte length of |bytecode_data|.
  uint32_t bytecode_length;
} iree_vm_bytecode_function_table_t;

// Mapped canonical constant pool.
typedef struct iree_vm_bytecode_constant_table_t {
  // Number of constant cells.
  uint32_t count;
  // Naturally aligned constant cells.
  const iree_vm_bytecode_v0_constant_cell_t* cells;
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
  // Number of distinct rodata blocks.
  uint32_t count;
  // Exact block lengths in ordinal order.
  const iree_vm_bytecode_v0_rodata_block_length_t* lengths;
  // First byte after the complete length array.
  const uint8_t* blocks_begin;
  // One-past-last byte of the rodata section.
  const uint8_t* section_end;
} iree_vm_bytecode_rodata_table_t;

// Mapped canonical authored presentation.
typedef struct iree_vm_bytecode_presentation_table_t {
  // Number of sparse declaration entries.
  uint32_t entry_count;
  // Declaration-sorted presentation rows.
  const iree_vm_bytecode_v0_presentation_entry_row_t* entries;
  // Number of signature-derived field rows.
  uint32_t field_count;
  // Argument-then-result field rows.
  const iree_vm_bytecode_v0_presentation_field_row_t* fields;
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

// Complete fixed mapped view of one verified module image.
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
  // First immutable value-global set bit word.
  iree_host_size_t value_set_bits_offset;
  // First immutable ref-global set bit word.
  iree_host_size_t ref_set_bits_offset;
  // First immutable function-global set bit word.
  iree_host_size_t function_set_bits_offset;
} iree_vm_bytecode_process_layout_t;

// Bounded allocation facts derived before constructing persistent storage.
typedef struct iree_vm_bytecode_module_plan_t {
  // Complete mapped image layout.
  iree_vm_bytecode_module_layout_t layout;
  // Exact process storage layout.
  iree_vm_bytecode_process_layout_t process_layout;
} iree_vm_bytecode_module_plan_t;

typedef struct iree_vm_bytecode_image_t iree_vm_bytecode_image_t;

// Mapped bytecode module embedded in its private image slab.
typedef struct iree_vm_bytecode_module_t {
  // Generic module provider base.
  iree_vm_module_t base;
  // Generic immutable module descriptor.
  iree_vm_module_descriptor_t descriptor;
  // Image owner containing this module.
  iree_vm_bytecode_image_t* image;
  // Verified mapped image layout.
  iree_vm_bytecode_module_layout_t layout;
  // Exact executable process storage layout, or zero for inspection.
  iree_vm_bytecode_process_layout_t process_layout;
  // Flat canonical resolved or reflection-only ref-type handles.
  iree_vm_ref_type_t* resolved_ref_types;
  // Canonical core vm.buffer descriptor used by executable rodata roots.
  iree_vm_ref_type_t buffer_type;
  // Embedded module-owned read-only executable rodata roots.
  iree_vm_buffer_t* rodata_roots;
} iree_vm_bytecode_module_t;

// Private image owner containing a module and all variable tails.
struct iree_vm_bytecode_image_t {
  // Owners held by the module and each embedded rodata root.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this complete image slab.
  iree_allocator_t host_allocator;
  // Transferred immutable image storage.
  iree_vm_bytecode_module_storage_t storage;
  // Embedded mapped module.
  iree_vm_bytecode_module_t module;
};

// Process-private construction state stored at offset zero.
typedef struct iree_vm_bytecode_process_state_t {
  // Open or sealed construction state.
  uint32_t construction_state;
  // Reserved zero word preserving natural cell alignment.
  uint32_t reserved;
} iree_vm_bytecode_process_state_t;

enum iree_vm_bytecode_construction_state_e {
  IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN = 0u,
  IREE_VM_BYTECODE_CONSTRUCTION_STATE_SEALED = 1u,
};

static inline iree_vm_bytecode_process_state_t* iree_vm_bytecode_process_state(
    void* process_storage) {
  return (iree_vm_bytecode_process_state_t*)process_storage;
}

static inline uint64_t* iree_vm_bytecode_process_values(
    const iree_vm_bytecode_module_t* module, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     module->process_layout.values_offset);
}

static inline iree_vm_ref_t* iree_vm_bytecode_process_refs(
    const iree_vm_bytecode_module_t* module, void* process_storage) {
  return (iree_vm_ref_t*)((uint8_t*)process_storage +
                          module->process_layout.refs_offset);
}

static inline iree_vm_function_ref_t* iree_vm_bytecode_process_functions(
    const iree_vm_bytecode_module_t* module, void* process_storage) {
  return (iree_vm_function_ref_t*)((uint8_t*)process_storage +
                                   module->process_layout.functions_offset);
}

static inline uint64_t* iree_vm_bytecode_process_value_set_bits(
    const iree_vm_bytecode_module_t* module, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     module->process_layout.value_set_bits_offset);
}

static inline uint64_t* iree_vm_bytecode_process_ref_set_bits(
    const iree_vm_bytecode_module_t* module, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     module->process_layout.ref_set_bits_offset);
}

static inline uint64_t* iree_vm_bytecode_process_function_set_bits(
    const iree_vm_bytecode_module_t* module, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     module->process_layout.function_set_bits_offset);
}

static inline bool iree_vm_bytecode_bit_test(const uint64_t* bits,
                                             uint32_t ordinal) {
  return (bits[ordinal / 64u] & ((uint64_t)1u << (ordinal % 64u))) != 0;
}

static inline void iree_vm_bytecode_bit_set(uint64_t* bits, uint32_t ordinal) {
  bits[ordinal / 64u] |= (uint64_t)1u << (ordinal % 64u);
}

static inline const iree_vm_bytecode_module_t*
iree_vm_bytecode_module_cast_const(const iree_vm_module_t* base) {
  return (const iree_vm_bytecode_module_t*)base;
}

static inline iree_vm_bytecode_module_t* iree_vm_bytecode_module_cast(
    iree_vm_module_t* base) {
  return (iree_vm_bytecode_module_t*)base;
}

#endif  // IREE_VM_BYTECODE_MODULE_STORAGE_H_
