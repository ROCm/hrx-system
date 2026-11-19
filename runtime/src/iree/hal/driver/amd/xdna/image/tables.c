// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/tables.h"

iree_status_t iree_hal_amd_xdna_image_tables_initialize(
    const iree_xdna_elf_header_record_t* header, uint32_t load_count,
    iree_allocator_t allocator, iree_hal_amd_xdna_image_tables_t* out_tables) {
  iree_host_size_t allocation_offset = 0;
  iree_host_size_t allocation_use_offset = 0;
  iree_host_size_t entry_offset = 0;
  iree_host_size_t binding_offset = 0;
  iree_host_size_t relocation_offset = 0;
  iree_host_size_t invocation_offset = 0;
  iree_host_size_t loads_offset = 0, strings_offset = 0, size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD_ALIGNED(
          header->allocation_count, iree_xdna_elf_allocation_record_t,
          iree_alignof(iree_xdna_elf_allocation_record_t), &allocation_offset),
      IREE_STRUCT_FIELD_ALIGNED(header->allocation_use_count, uint32_t,
                                iree_alignof(uint32_t), &allocation_use_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          header->entry_count, iree_xdna_elf_entry_record_t,
          iree_alignof(iree_xdna_elf_entry_record_t), &entry_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          header->binding_count, iree_xdna_elf_binding_record_t,
          iree_alignof(iree_xdna_elf_binding_record_t), &binding_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          header->relocation_count, iree_xdna_elf_relocation_record_t,
          iree_alignof(iree_xdna_elf_relocation_record_t), &relocation_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          header->invocation_count, iree_xdna_elf_invocation_record_t,
          iree_alignof(iree_xdna_elf_invocation_record_t), &invocation_offset),
      IREE_STRUCT_FIELD_ALIGNED(load_count, iree_hal_amd_xdna_image_load_t,
                                iree_alignof(iree_hal_amd_xdna_image_load_t),
                                &loads_offset),
      IREE_STRUCT_FIELD(header->string_byte_length, char, &strings_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, size, (void**)&storage));
  *out_tables = (iree_hal_amd_xdna_image_tables_t){
      .header = *header,
      .allocations =
          (iree_xdna_elf_allocation_record_t*)(storage + allocation_offset),
      .allocation_uses = (uint32_t*)(storage + allocation_use_offset),
      .entries = (iree_xdna_elf_entry_record_t*)(storage + entry_offset),
      .bindings = (iree_xdna_elf_binding_record_t*)(storage + binding_offset),
      .relocations =
          (iree_xdna_elf_relocation_record_t*)(storage + relocation_offset),
      .invocations =
          (iree_xdna_elf_invocation_record_t*)(storage + invocation_offset),
      .loads = (iree_hal_amd_xdna_image_load_t*)(storage + loads_offset),
      .strings = (char*)storage + strings_offset,
  };
  return iree_ok_status();
}

void iree_hal_amd_xdna_image_tables_deinitialize(
    iree_hal_amd_xdna_image_tables_t* tables, iree_allocator_t allocator) {
  // The allocation table begins at offset zero of the joint allocation.
  iree_allocator_free(allocator, tables->allocations);
  *tables = (iree_hal_amd_xdna_image_tables_t){0};
}

iree_status_t iree_hal_amd_xdna_image_tables_decode(
    iree_const_byte_span_t storage, uint32_t load_count,
    iree_allocator_t allocator, iree_hal_amd_xdna_image_tables_t* out_tables) {
  if (storage.data_length < IREE_XDNA_ELF_HEADER_RECORD_SIZE ||
      storage.data_length > IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid XDNA metadata extent");
  }
  const iree_xdna_elf_header_record_t header =
      iree_xdna_elf_decode_header(storage.data);
  if (header.magic != IREE_XDNA_ELF_METADATA_MAGIC ||
      header.version != IREE_XDNA_ELF_METADATA_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA metadata ABI");
  }
  if (header.allocation_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA allocation count exceeds image limits");
  }
  if (header.allocation_use_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA allocation use count exceeds image limits");
  }
  if (header.entry_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA entry count exceeds image limits");
  }
  if (header.binding_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding count exceeds image limits");
  }
  if (header.relocation_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA relocation count exceeds image limits");
  }
  if (header.invocation_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA invocation count exceeds image limits");
  }
  // The count limits above bound the sum of the fixed-width wire tables.
  const uint32_t offset =
      IREE_XDNA_ELF_HEADER_RECORD_SIZE +
      header.allocation_count * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE +
      header.allocation_use_count * sizeof(uint32_t) +
      header.entry_count * IREE_XDNA_ELF_ENTRY_RECORD_SIZE +
      header.binding_count * IREE_XDNA_ELF_BINDING_RECORD_SIZE +
      header.relocation_count * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE +
      header.invocation_count * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE;
  if (offset > storage.data_length ||
      header.string_byte_length != storage.data_length - offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA metadata tables do not match their extent");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_tables_initialize(
      &header, load_count, allocator, out_tables));
  const uint8_t* row = storage.data + IREE_XDNA_ELF_HEADER_RECORD_SIZE;
  for (uint32_t i = 0; i < header.allocation_count;
       ++i, row += IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE) {
    out_tables->allocations[i] = iree_xdna_elf_decode_allocation(row);
  }
  for (uint32_t i = 0; i < header.allocation_use_count;
       ++i, row += sizeof(uint32_t)) {
    out_tables->allocation_uses[i] = iree_unaligned_load_le_u32(row);
  }
  for (uint32_t i = 0; i < header.entry_count;
       ++i, row += IREE_XDNA_ELF_ENTRY_RECORD_SIZE) {
    out_tables->entries[i] = iree_xdna_elf_decode_entry(row);
  }
  for (uint32_t i = 0; i < header.binding_count;
       ++i, row += IREE_XDNA_ELF_BINDING_RECORD_SIZE) {
    out_tables->bindings[i] = iree_xdna_elf_decode_binding(row);
  }
  for (uint32_t i = 0; i < header.relocation_count;
       ++i, row += IREE_XDNA_ELF_RELOCATION_RECORD_SIZE) {
    out_tables->relocations[i] = iree_xdna_elf_decode_relocation(row);
  }
  for (uint32_t i = 0; i < header.invocation_count;
       ++i, row += IREE_XDNA_ELF_INVOCATION_RECORD_SIZE) {
    out_tables->invocations[i] = iree_xdna_elf_decode_invocation(row);
  }
  memcpy(out_tables->strings, row, header.string_byte_length);
  return iree_ok_status();
}
