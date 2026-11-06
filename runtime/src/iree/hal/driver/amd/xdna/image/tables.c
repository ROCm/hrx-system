// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/tables.h"

iree_status_t iree_hal_amd_xdna_image_tables_initialize(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_image_tables_t* out_tables) {
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
  // Bounded counts make the fixed table products representable in uint32_t.
  out_tables->storage = storage;
  out_tables->header = header;
  uint32_t offset = IREE_XDNA_ELF_HEADER_RECORD_SIZE;
  out_tables->allocation_offset = offset;
  offset += header.allocation_count * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE;
  out_tables->allocation_use_offset = offset;
  offset += header.allocation_use_count * sizeof(uint32_t);
  out_tables->entry_offset = offset;
  offset += header.entry_count * IREE_XDNA_ELF_ENTRY_RECORD_SIZE;
  out_tables->binding_offset = offset;
  offset += header.binding_count * IREE_XDNA_ELF_BINDING_RECORD_SIZE;
  out_tables->relocation_offset = offset;
  offset += header.relocation_count * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE;
  out_tables->invocation_offset = offset;
  offset += header.invocation_count * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE;
  out_tables->string_offset = offset;
  if (offset > storage.data_length ||
      header.string_byte_length != storage.data_length - offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA metadata tables do not match their extent");
  }
  return iree_ok_status();
}
