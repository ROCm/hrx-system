// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/hsaco_elf.h"

#include <string.h>

#define IREE_HAL_AMDGPU_ELF64_HEADER_SIZE 64
#define IREE_HAL_AMDGPU_ELF64_SECTION_HEADER_SIZE 64
#define IREE_HAL_AMDGPU_ELF64_SYMBOL_SIZE 24

static bool iree_hal_amdgpu_hsaco_elf_u64_to_host_size(
    uint64_t value, iree_host_size_t* out_value) {
  if (value > (uint64_t)IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_value = (iree_host_size_t)value;
  return true;
}

static bool iree_hal_amdgpu_hsaco_elf_section_range(
    iree_const_byte_span_t elf_data, const uint8_t* section,
    iree_host_size_t* out_offset, iree_host_size_t* out_size) {
  *out_offset = 0;
  *out_size = 0;
  const uint64_t offset_u64 = iree_unaligned_load_le_u64(section + 24);
  const uint64_t size_u64 = iree_unaligned_load_le_u64(section + 32);
  if (offset_u64 > elf_data.data_length ||
      size_u64 > elf_data.data_length - offset_u64) {
    return false;
  }
  return iree_hal_amdgpu_hsaco_elf_u64_to_host_size(offset_u64, out_offset) &&
         iree_hal_amdgpu_hsaco_elf_u64_to_host_size(size_u64, out_size);
}

bool iree_hal_amdgpu_hsaco_elf_section(iree_const_byte_span_t elf_data,
                                       uint16_t section_index,
                                       const uint8_t** out_section) {
  *out_section = NULL;
  if (elf_data.data_length < IREE_HAL_AMDGPU_ELF64_HEADER_SIZE) {
    return false;
  }
  const uint8_t* header = elf_data.data;
  const uint64_t section_offset = iree_unaligned_load_le_u64(header + 40);
  const uint16_t section_entry_size = iree_unaligned_load_le_u16(header + 58);
  const uint16_t section_count = iree_unaligned_load_le_u16(header + 60);
  if (section_index >= section_count ||
      section_entry_size < IREE_HAL_AMDGPU_ELF64_SECTION_HEADER_SIZE) {
    return false;
  }
  iree_host_size_t section_offset_host = 0;
  if (!iree_hal_amdgpu_hsaco_elf_u64_to_host_size(section_offset,
                                                  &section_offset_host)) {
    return false;
  }
  iree_host_size_t section_relative_offset = 0;
  iree_host_size_t selected_offset = 0;
  if (!iree_host_size_checked_mul(section_index, section_entry_size,
                                  &section_relative_offset) ||
      !iree_host_size_checked_add(section_offset_host, section_relative_offset,
                                  &selected_offset) ||
      selected_offset > elf_data.data_length ||
      IREE_HAL_AMDGPU_ELF64_SECTION_HEADER_SIZE >
          elf_data.data_length - selected_offset) {
    return false;
  }
  *out_section = elf_data.data + selected_offset;
  return true;
}

bool iree_hal_amdgpu_hsaco_elf_symbol_table_initialize(
    iree_const_byte_span_t elf_data, const uint8_t* section,
    iree_hal_amdgpu_hsaco_elf_symbol_table_t* out_symbol_table) {
  memset(out_symbol_table, 0, sizeof(*out_symbol_table));
  iree_host_size_t section_size = 0;
  if (!iree_hal_amdgpu_hsaco_elf_section_range(
          elf_data, section, &out_symbol_table->offset, &section_size)) {
    return false;
  }
  const uint64_t entry_size_u64 = iree_unaligned_load_le_u64(section + 56);
  if (entry_size_u64 < IREE_HAL_AMDGPU_ELF64_SYMBOL_SIZE ||
      !iree_hal_amdgpu_hsaco_elf_u64_to_host_size(
          entry_size_u64, &out_symbol_table->entry_size)) {
    return false;
  }
  out_symbol_table->section = section;
  out_symbol_table->count = section_size / out_symbol_table->entry_size;
  return true;
}

bool iree_hal_amdgpu_hsaco_elf_symbol(
    iree_const_byte_span_t elf_data,
    const iree_hal_amdgpu_hsaco_elf_symbol_table_t* symbol_table,
    iree_host_size_t symbol_index, const uint8_t** out_symbol) {
  *out_symbol = NULL;
  iree_host_size_t relative_offset = 0;
  iree_host_size_t symbol_offset = 0;
  if (symbol_index >= symbol_table->count ||
      !iree_host_size_checked_mul(symbol_index, symbol_table->entry_size,
                                  &relative_offset) ||
      !iree_host_size_checked_add(symbol_table->offset, relative_offset,
                                  &symbol_offset) ||
      symbol_offset > elf_data.data_length ||
      IREE_HAL_AMDGPU_ELF64_SYMBOL_SIZE >
          elf_data.data_length - symbol_offset) {
    return false;
  }
  *out_symbol = elf_data.data + symbol_offset;
  return true;
}

iree_string_view_t iree_hal_amdgpu_hsaco_elf_symbol_name(
    iree_const_byte_span_t elf_data,
    const iree_hal_amdgpu_hsaco_elf_symbol_table_t* symbol_table,
    const uint8_t* symbol) {
  const uint32_t string_section_index =
      iree_unaligned_load_le_u32(symbol_table->section + 40);
  if (string_section_index > UINT16_MAX) {
    return iree_string_view_empty();
  }
  const uint8_t* string_section = NULL;
  if (!iree_hal_amdgpu_hsaco_elf_section(
          elf_data, (uint16_t)string_section_index, &string_section)) {
    return iree_string_view_empty();
  }
  iree_host_size_t string_offset = 0;
  iree_host_size_t string_size = 0;
  const uint32_t name_offset = iree_unaligned_load_le_u32(symbol);
  if (!iree_hal_amdgpu_hsaco_elf_section_range(elf_data, string_section,
                                               &string_offset, &string_size) ||
      name_offset >= string_size) {
    return iree_string_view_empty();
  }
  const char* name = (const char*)elf_data.data + string_offset + name_offset;
  const char* end = (const char*)elf_data.data + string_offset + string_size;
  const char* p = name;
  while (p < end && *p) {
    ++p;
  }
  return iree_make_string_view(name, (iree_host_size_t)(p - name));
}
