// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/aie_sections.h"

#include <string.h>

#include "iree/hal/driver/amd/xdna/image/aie_full_elf.h"
#include "iree/schemas/xdna_executable.h"

static iree_status_t iree_hal_amd_aie_string(
    const iree_hal_amd_aie_sections_t* file, uint32_t table, uint32_t offset,
    iree_string_view_t* out_name) {
  if (table >= file->count ||
      file->sections[table].type != IREE_HAL_AMD_AIE_STRTAB ||
      offset >= file->sections[table].size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE ELF string reference");
  }
  const iree_hal_amd_aie_section_t* section = &file->sections[table];
  const char* begin = (const char*)section->records + offset;
  const char* end = memchr(begin, 0, section->size - offset);
  if (!end) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated AIE ELF string");
  }
  *out_name = iree_make_string_view(begin, end - begin);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_aie_read_section(
    iree_hal_amd_aie_sections_t* file, const uint8_t* row, uint64_t length,
    uint64_t* metadata_bytes, iree_hal_amd_aie_section_t* section) {
  section->name_offset = iree_unaligned_load_le_u32(row);
  section->type = iree_unaligned_load_le_u32(row + 4);
  section->flags = iree_unaligned_load_le_u32(row + 8);
  section->offset = iree_unaligned_load_le_u32(row + 16);
  section->size = iree_unaligned_load_le_u32(row + 20);
  section->link = iree_unaligned_load_le_u32(row + 24);
  section->info = iree_unaligned_load_le_u32(row + 28);
  section->alignment = iree_unaligned_load_le_u32(row + 32);
  if (section->type != 8 /* SHT_NOBITS */ &&
      (uint64_t)section->offset + section->size > length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "truncated AIE ELF section");
  }
  uint32_t stride = 0;
  switch (section->type) {
    case IREE_HAL_AMD_AIE_GROUP:
      stride = 4;
      break;
    case IREE_HAL_AMD_AIE_SYMTAB:
    case IREE_HAL_AMD_AIE_DYNSYM:
      stride = 16;
      break;
    case IREE_HAL_AMD_AIE_RELA:
      stride = 12;
      break;
    case IREE_HAL_AMD_AIE_STRTAB:
    case IREE_HAL_AMD_AIE_NOTE:
      break;
    case 9:  // SHT_REL
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "AIE ELF requires RELA relocations");
    default:
      return iree_ok_status();
  }
  if (section->flags & 0x800 /* SHF_COMPRESSED */) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "compressed AIE ELF metadata");
  }
  if (stride && (iree_unaligned_load_le_u32(row + 36) != stride ||
                 section->size % stride != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE ELF record stride");
  }
  *metadata_bytes += section->size;
  if (*metadata_bytes > IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE ELF metadata exceeds image limits");
  }
  if (file->contents.data) {
    section->records = file->contents.data + section->offset;
  } else if (section->size) {
    uint8_t* copy = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        file->allocator, section->size, (void**)&copy));
    section->records = copy;
    IREE_RETURN_IF_ERROR(
        iree_byte_sequence_read(file->source, section->offset,
                                iree_make_byte_span(copy, section->size)));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_aie_sections_initialize(
    iree_byte_sequence_t* source, iree_allocator_t allocator,
    iree_hal_amd_aie_sections_t* out_file) {
  iree_hal_amd_aie_sections_t file = {.allocator = allocator, .source = source};
  iree_byte_sequence_try_get_contiguous_span(source, &file.contents);
  uint8_t header[IREE_XDNA_ELF_HEADER_SIZE];
  IREE_RETURN_IF_ERROR(iree_byte_sequence_read(
      source, 0, iree_make_byte_span(header, sizeof(header))));
  static const uint8_t identity[] = {0x7f, 'E', 'L', 'F',
                                     1,    1,   1,   IREE_HAL_AMD_AIE_OS_ABI};
  if (memcmp(header, identity, sizeof(identity)) ||
      header[8] != IREE_HAL_AMD_AIE_GROUP_ABI ||
      iree_unaligned_load_le_u16(header + 16) != 2 /* ET_EXEC */ ||
      iree_unaligned_load_le_u16(header + 18) != 1 /* AIEBU uses EM_M32 */ ||
      iree_unaligned_load_le_u32(header + 20) != 1 ||
      iree_unaligned_load_le_u16(header + 40) != sizeof(header) ||
      iree_unaligned_load_le_u16(header + 46) !=
          IREE_XDNA_ELF_SECTION_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a grouped AIE2P ELF32LE executable");
  }
  const uint32_t offset = iree_unaligned_load_le_u32(header + 32);
  const uint32_t count = iree_unaligned_load_le_u16(header + 48);
  const uint32_t names = iree_unaligned_load_le_u16(header + 50);
  const uint64_t source_length = iree_byte_sequence_length(source);
  if (count == 0 || count > IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT ||
      names >= count || offset < sizeof(header) ||
      (uint64_t)offset + count * IREE_XDNA_ELF_SECTION_HEADER_SIZE >
          source_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE ELF section directory");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, count * sizeof(*file.sections), (void**)&file.sections));
  file.count = count;
  iree_status_t status = iree_ok_status();
  uint64_t metadata_bytes = 0;
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    uint8_t row[IREE_XDNA_ELF_SECTION_HEADER_SIZE];
    status = iree_byte_sequence_read(source, (uint64_t)offset + i * sizeof(row),
                                     iree_make_byte_span(row, sizeof(row)));
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_aie_read_section(
          &file, row, source_length, &metadata_bytes, &file.sections[i]);
    }
  }
  if (iree_status_is_ok(status) && file.sections[0].type != 0 /* SHT_NULL */) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected null AIE ELF section zero");
  }
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_aie_string(&file, names, file.sections[i].name_offset,
                                     &file.sections[i].name);
  }
  if (iree_status_is_ok(status)) {
    *out_file = file;
  } else {
    iree_hal_amd_aie_sections_deinitialize(&file);
  }
  return status;
}

void iree_hal_amd_aie_sections_deinitialize(iree_hal_amd_aie_sections_t* file) {
  if (!file->contents.data) {
    for (uint32_t i = 0; i < file->count; ++i) {
      iree_allocator_free(file->allocator, (void*)file->sections[i].records);
    }
  }
  iree_allocator_free(file->allocator, file->sections);
}

iree_status_t iree_hal_amd_aie_sections_symbol(
    const iree_hal_amd_aie_sections_t* file, uint32_t table,
    uint32_t table_type, uint32_t ordinal, uint8_t symbol_type,
    iree_string_view_t* out_name, const uint8_t** out_record) {
  if (table >= file->count || file->sections[table].type != table_type ||
      ordinal >= file->sections[table].size / 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE ELF symbol reference");
  }
  const uint8_t* symbol = file->sections[table].records + 16 * ordinal;
  if ((symbol[12] & 15) != symbol_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected AIE ELF symbol type");
  }
  *out_record = symbol;
  return iree_hal_amd_aie_string(file, file->sections[table].link,
                                 iree_unaligned_load_le_u32(symbol), out_name);
}
