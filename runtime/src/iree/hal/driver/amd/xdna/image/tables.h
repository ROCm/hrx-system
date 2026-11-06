// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Indexed, immutable metadata retained in its compact wire representation.

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TABLES_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TABLES_H_

#include "iree/schemas/xdna_executable.h"

#ifdef __cplusplus
extern "C" {
#endif

// Metadata view whose row boundaries have been established at admission.
// Rows remain little endian; queries load fixed fields without allocating or
// reconstructing relations. The storage owner outlives this borrowed view.
typedef struct iree_hal_amd_xdna_image_tables_t {
  // Complete metadata bytes, borrowed from the image.
  iree_const_byte_span_t storage;
  // Decoded target requirements and table cardinalities.
  iree_xdna_elf_header_record_t header;
  // Byte offset of the allocation table.
  uint32_t allocation_offset;
  // Byte offset of the allocation use table.
  uint32_t allocation_use_offset;
  // Byte offset of the entry table.
  uint32_t entry_offset;
  // Byte offset of the binding table.
  uint32_t binding_offset;
  // Byte offset of the relocation table.
  uint32_t relocation_offset;
  // Byte offset of the invocation table.
  uint32_t invocation_offset;
  // Byte offset of the trailing name table.
  uint32_t string_offset;
} iree_hal_amd_xdna_image_tables_t;

// Establishes all row boundaries from the untrusted header and exact extent.
// Row relationships and target admission are validated by image_create.
iree_status_t iree_hal_amd_xdna_image_tables_initialize(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_image_tables_t* out_tables);

// Returns the row at a valid allocation-table ordinal.
static inline iree_xdna_elf_allocation_record_t
iree_hal_amd_xdna_image_tables_allocation(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return iree_xdna_elf_decode_allocation(
      tables->storage.data + tables->allocation_offset +
      ordinal * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE);
}

// Returns the global allocation index at a valid use-table ordinal.
static inline uint32_t iree_hal_amd_xdna_image_tables_allocation_use(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return iree_unaligned_load_le_u32(tables->storage.data +
                                    tables->allocation_use_offset +
                                    ordinal * sizeof(uint32_t));
}

// Returns the row at a valid entry-table ordinal.
static inline iree_xdna_elf_entry_record_t iree_hal_amd_xdna_image_tables_entry(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return iree_xdna_elf_decode_entry(tables->storage.data +
                                    tables->entry_offset +
                                    ordinal * IREE_XDNA_ELF_ENTRY_RECORD_SIZE);
}

// Returns the row at a valid binding-table ordinal.
static inline iree_xdna_elf_binding_record_t
iree_hal_amd_xdna_image_tables_binding(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return iree_xdna_elf_decode_binding(
      tables->storage.data + tables->binding_offset +
      ordinal * IREE_XDNA_ELF_BINDING_RECORD_SIZE);
}

// Returns the row at a valid relocation-table ordinal.
static inline iree_xdna_elf_relocation_record_t
iree_hal_amd_xdna_image_tables_relocation(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return iree_xdna_elf_decode_relocation(
      tables->storage.data + tables->relocation_offset +
      ordinal * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE);
}

// Returns the row at a valid invocation-table ordinal.
static inline iree_xdna_elf_invocation_record_t
iree_hal_amd_xdna_image_tables_invocation(
    const iree_hal_amd_xdna_image_tables_t* tables, uint32_t ordinal) {
  return iree_xdna_elf_decode_invocation(
      tables->storage.data + tables->invocation_offset +
      ordinal * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE);
}

// Returns an admitted entry's name, borrowing the table storage.
static inline iree_string_view_t iree_hal_amd_xdna_image_tables_entry_name(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_xdna_elf_entry_record_t* entry) {
  return iree_make_string_view((const char*)tables->storage.data +
                                   tables->string_offset + entry->name_offset,
                               entry->name_length);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TABLES_H_
