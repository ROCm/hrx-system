// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Indexed execution metadata shared by the XDNA executable readers.

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TABLES_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TABLES_H_

#include "iree/schemas/xdna_executable.h"

#ifdef __cplusplus
extern "C" {
#endif

// Standalone controller field naming a PDI in firmware address space. Native
// .xdna metadata defines only SHIM_ADDRESS and cannot declare this kind.
enum { IREE_HAL_AMD_XDNA_RELOCATION_PDI_ADDRESS_64 = 2 };

// A source range copied into an allocation when its backing is materialized.
typedef struct iree_hal_amd_xdna_image_load_t {
  // First byte in the retained executable.
  uint32_t source_offset;
  // First byte in the destination allocation.
  uint32_t destination_offset;
  // Number of source bytes copied.
  uint32_t file_size;
  // Extent initialized, including the zero-filled tail.
  uint32_t memory_size;
} iree_hal_amd_xdna_image_load_t;

// Readers populate these rows once during admission. The executable owns one
// aligned host allocation for the rows, payload ranges, and export names.
typedef struct iree_hal_amd_xdna_image_tables_t {
  // Requirements and cardinalities established by the reader.
  iree_xdna_elf_header_record_t header;
  // Allocation requirements in image order.
  iree_xdna_elf_allocation_record_t* allocations;
  // Global allocation ordinal for each entry-relative use.
  uint32_t* allocation_uses;
  // Exported entries in reflection order.
  iree_xdna_elf_entry_record_t* entries;
  // External buffer contracts in entry-relative binding order.
  iree_xdna_elf_binding_record_t* bindings;
  // Address fields grouped by entry and binding time.
  iree_xdna_elf_relocation_record_t* relocations;
  // Native command ranges for each entry.
  iree_xdna_elf_invocation_record_t* invocations;
  // Initialization ranges addressed by each allocation's first_load.
  iree_hal_amd_xdna_image_load_t* loads;
  // Concatenated export names addressed by each entry's name_offset.
  char* strings;
} iree_hal_amd_xdna_image_tables_t;

// Allocates zero-initialized rows and strings at checked offsets.
iree_status_t iree_hal_amd_xdna_image_tables_initialize(
    const iree_xdna_elf_header_record_t* header, uint32_t load_count,
    iree_allocator_t allocator, iree_hal_amd_xdna_image_tables_t* out_tables);

// Decodes a native metadata segment into aligned host rows. Load ranges are
// populated from the corresponding ELF program headers after validation.
iree_status_t iree_hal_amd_xdna_image_tables_decode(
    iree_const_byte_span_t storage, uint32_t load_count,
    iree_allocator_t allocator, iree_hal_amd_xdna_image_tables_t* out_tables);

// Releases storage allocated by either reader. Zero initialization is valid.
void iree_hal_amd_xdna_image_tables_deinitialize(
    iree_hal_amd_xdna_image_tables_t* tables, iree_allocator_t allocator);

// Allocation requirements in image order.
static inline iree_xdna_elf_allocation_record_t
iree_hal_amd_xdna_image_tables_allocation(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return tables->allocations[ordinal];
}

// Global allocation ordinal for each entry-relative use.
static inline uint32_t iree_hal_amd_xdna_image_tables_allocation_use(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return tables->allocation_uses[ordinal];
}

// Exported entries in reflection order.
static inline iree_xdna_elf_entry_record_t iree_hal_amd_xdna_image_tables_entry(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return tables->entries[ordinal];
}

// External buffer contracts in entry-relative binding order.
static inline iree_xdna_elf_binding_record_t
iree_hal_amd_xdna_image_tables_binding(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return tables->bindings[ordinal];
}

// Address fields grouped by entry and binding time.
static inline iree_xdna_elf_relocation_record_t
iree_hal_amd_xdna_image_tables_relocation(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return tables->relocations[ordinal];
}

// Native command ranges for each entry.
static inline iree_xdna_elf_invocation_record_t
iree_hal_amd_xdna_image_tables_invocation(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return tables->invocations[ordinal];
}

// Returns an admitted entry's name, borrowing the table storage.
static inline iree_string_view_t iree_hal_amd_xdna_image_tables_entry_name(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_xdna_elf_entry_record_t* entry) {
  return iree_make_string_view(tables->strings + entry->name_offset,
                               entry->name_length);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TABLES_H_
