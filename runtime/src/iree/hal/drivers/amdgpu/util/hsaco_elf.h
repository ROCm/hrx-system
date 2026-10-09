// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_ELF_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_ELF_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Bounds-checked view of one ELF symbol table.
typedef struct iree_hal_amdgpu_hsaco_elf_symbol_table_t {
  // Symbol section header borrowed from the ELF data.
  const uint8_t* section;
  // Byte offset of the first symbol in the ELF data.
  iree_host_size_t offset;
  // Byte stride between symbols.
  iree_host_size_t entry_size;
  // Number of complete symbols in the section.
  iree_host_size_t count;
} iree_hal_amdgpu_hsaco_elf_symbol_table_t;

// Returns a bounds-checked section header borrowed from |elf_data|.
bool iree_hal_amdgpu_hsaco_elf_section(iree_const_byte_span_t elf_data,
                                       uint16_t section_index,
                                       const uint8_t** out_section);

// Initializes a bounds-checked symbol-table view for |section|.
bool iree_hal_amdgpu_hsaco_elf_symbol_table_initialize(
    iree_const_byte_span_t elf_data, const uint8_t* section,
    iree_hal_amdgpu_hsaco_elf_symbol_table_t* out_symbol_table);

// Returns a bounds-checked symbol borrowed from |elf_data|.
bool iree_hal_amdgpu_hsaco_elf_symbol(
    iree_const_byte_span_t elf_data,
    const iree_hal_amdgpu_hsaco_elf_symbol_table_t* symbol_table,
    iree_host_size_t symbol_index, const uint8_t** out_symbol);

// Returns the bounded name of |symbol| or an empty view if malformed.
iree_string_view_t iree_hal_amdgpu_hsaco_elf_symbol_name(
    iree_const_byte_span_t elf_data,
    const iree_hal_amdgpu_hsaco_elf_symbol_table_t* symbol_table,
    const uint8_t* symbol);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_ELF_H_
