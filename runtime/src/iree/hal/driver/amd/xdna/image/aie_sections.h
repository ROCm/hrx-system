// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE_SECTIONS_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE_SECTIONS_H_

#include "iree/base/byte_sequence.h"

enum {
  IREE_HAL_AMD_AIE_GROUP_ABI = 0x21,
  IREE_HAL_AMD_AIE_PROGBITS = 1,
  IREE_HAL_AMD_AIE_SYMTAB = 2,
  IREE_HAL_AMD_AIE_STRTAB = 3,
  IREE_HAL_AMD_AIE_RELA = 4,
  IREE_HAL_AMD_AIE_NOTE = 7,
  IREE_HAL_AMD_AIE_DYNSYM = 11,
  IREE_HAL_AMD_AIE_GROUP = 17,
};

// Decoded ELF32 section header and its bounded metadata, when applicable.
typedef struct iree_hal_amd_aie_section_t {
  // Identifier borrowed from the section name table.
  iree_string_view_t name;
  // Byte offset of the name in the section string table.
  uint32_t name_offset;
  // ELF section type.
  uint32_t type;
  // ELF section flags.
  uint32_t flags;
  // File offset of the payload.
  uint32_t offset;
  // Section extent in bytes.
  uint32_t size;
  // Linked symbol or string table ordinal.
  uint32_t link;
  // Group signature symbol ordinal for SHT_GROUP.
  uint32_t info;
  // Required section alignment in bytes.
  uint32_t alignment;
  // Metadata borrowed from the source or owned by the section directory.
  // Program and PDI contents remain in the source.
  const uint8_t* records;
} iree_hal_amd_aie_section_t;

// Load-time view of a modern AIEBU section directory.
typedef struct iree_hal_amd_aie_sections_t {
  // Allocator owning the directory and any segmented metadata copies.
  iree_allocator_t allocator;
  // Source borrowed throughout admission.
  iree_byte_sequence_t* source;
  // Contiguous source bytes when available.
  iree_const_byte_span_t contents;
  // Number of section headers.
  uint32_t count;
  // Headers in file ordinal order.
  iree_hal_amd_aie_section_t* sections;
} iree_hal_amd_aie_sections_t;

// Reads section, group, string, and relocation metadata. Deinitialize after
// success.
iree_status_t iree_hal_amd_aie_sections_initialize(
    iree_byte_sequence_t* source, iree_allocator_t allocator,
    iree_hal_amd_aie_sections_t* out_file);
void iree_hal_amd_aie_sections_deinitialize(iree_hal_amd_aie_sections_t* file);

// Returns the symbol's bounded name and ELF32 record, checking the table,
// ordinal, string reference, and required ELF symbol type.
iree_status_t iree_hal_amd_aie_sections_symbol(
    const iree_hal_amd_aie_sections_t* file, uint32_t table,
    uint32_t table_type, uint32_t ordinal, uint8_t symbol_type,
    iree_string_view_t* out_name, const uint8_t** out_record);

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE_SECTIONS_H_
