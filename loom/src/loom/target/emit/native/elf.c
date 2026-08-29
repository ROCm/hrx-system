// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf.h"

#include <inttypes.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// ELF little-endian format records
//===----------------------------------------------------------------------===//

#define LOOM_NATIVE_ELF32LE_EHDR_SIZE 52u
#define LOOM_NATIVE_ELF32LE_PHDR_SIZE 32u
#define LOOM_NATIVE_ELF32LE_SHDR_SIZE 40u

#define LOOM_NATIVE_ELF64LE_EHDR_SIZE 64u
#define LOOM_NATIVE_ELF64LE_PHDR_SIZE 56u
#define LOOM_NATIVE_ELF64LE_SHDR_SIZE 64u

#define LOOM_NATIVE_ELF_EI_CLASS 4u
#define LOOM_NATIVE_ELF_EI_DATA 5u
#define LOOM_NATIVE_ELF_EI_VERSION 6u
#define LOOM_NATIVE_ELF_EI_OSABI 7u
#define LOOM_NATIVE_ELF_EI_ABIVERSION 8u
#define LOOM_NATIVE_ELF_EI_NIDENT 16u

#define LOOM_NATIVE_ELF_CLASS_32 1u
#define LOOM_NATIVE_ELF_CLASS_64 2u
#define LOOM_NATIVE_ELF_DATA_LITTLE_ENDIAN 1u
#define LOOM_NATIVE_ELF_VERSION_CURRENT 1u

typedef struct loom_native_elf_format_t {
  // ELFCLASS* value written to e_ident.
  uint8_t elf_class;
  // Serialized ELF header byte size.
  uint16_t header_size;
  // Serialized program-header byte size.
  uint16_t program_header_size;
  // Serialized section-header byte size.
  uint16_t section_header_size;
  // Alignment of the section-header table in the output file.
  uint8_t section_header_alignment;
} loom_native_elf_format_t;

static const loom_native_elf_format_t kLoomNativeElf32LeFormat = {
    .elf_class = LOOM_NATIVE_ELF_CLASS_32,
    .header_size = LOOM_NATIVE_ELF32LE_EHDR_SIZE,
    .program_header_size = LOOM_NATIVE_ELF32LE_PHDR_SIZE,
    .section_header_size = LOOM_NATIVE_ELF32LE_SHDR_SIZE,
    .section_header_alignment = 4,
};

static const loom_native_elf_format_t kLoomNativeElf64LeFormat = {
    .elf_class = LOOM_NATIVE_ELF_CLASS_64,
    .header_size = LOOM_NATIVE_ELF64LE_EHDR_SIZE,
    .program_header_size = LOOM_NATIVE_ELF64LE_PHDR_SIZE,
    .section_header_size = LOOM_NATIVE_ELF64LE_SHDR_SIZE,
    .section_header_alignment = 8,
};

// Class-neutral view over a public ELF32LE or ELF64LE file record.
typedef struct loom_native_elf_file_view_t {
  uint16_t type;
  uint16_t machine;
  uint8_t os_abi;
  uint8_t abi_version;
  uint32_t flags;
  uint64_t entry;
  const loom_native_elf_section_t* sections;
  iree_host_size_t section_count;
  const loom_native_elf_segment_t* segments;
  iree_host_size_t segment_count;
} loom_native_elf_file_view_t;

//===----------------------------------------------------------------------===//
// Layout model
//===----------------------------------------------------------------------===//

typedef struct loom_native_elf_section_layout_t {
  // Section-name offset within the generated `.shstrtab`.
  uint32_t name_offset;
  // Byte offset of the section contents in the output file.
  uint64_t file_offset;
  // Byte length of the section contents in the output file.
  uint64_t file_size;
  // Normalized power-of-two section alignment in bytes.
  uint64_t alignment;
} loom_native_elf_section_layout_t;

typedef struct loom_native_elf_layout_t {
  // Section layouts, including the null section and generated `.shstrtab`.
  loom_native_elf_section_layout_t* sections;
  // Number of entries in |sections|.
  iree_host_size_t section_count;
  // Arena-backed section-name string table bytes.
  iree_string_view_t string_table;
  // Byte offset of the ELF program-header table, or zero when absent.
  uint64_t program_header_offset;
  // Byte offset of the ELF section-header table.
  uint64_t section_header_offset;
  // Complete serialized file byte size.
  uint64_t file_size;
} loom_native_elf_layout_t;

static iree_status_t loom_native_elf_validate_alignment(
    uint64_t alignment, iree_string_view_t field_name) {
  if (alignment == 0) {
    return iree_ok_status();
  }
  if (!iree_is_power_of_two_uint64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF field `%.*s` alignment must be a power of two",
                            (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf_validate_section_name(
    iree_string_view_t name, iree_host_size_t index) {
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF section %" PRIhsz " name is required", index);
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    if (name.data[i] == '\0') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "ELF section %" PRIhsz " name contains an embedded NUL", index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf_validate_file(
    const loom_native_elf_file_view_t* file) {
  if (file->type == LOOM_NATIVE_ELF_FILE_TYPE_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF file type is required");
  }
  if (file->machine == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF machine is required");
  }
  if (file->section_count > UINT16_MAX - 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section count exceeds header capacity");
  }
  if (file->section_count != 0 && file->sections == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF sections are required for a nonzero count");
  }
  if (file->segment_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF segment count exceeds header capacity");
  }
  if (file->segment_count != 0 && file->segments == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF segments are required for a nonzero count");
  }

  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    const loom_native_elf_section_t* section = &file->sections[i];
    IREE_RETURN_IF_ERROR(
        loom_native_elf_validate_section_name(section->name, i));
    if (iree_string_view_equal(section->name, IREE_SV(".shstrtab"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "ELF section %" PRIhsz " uses the reserved `.shstrtab` name", i);
    }
    if (section->type == LOOM_NATIVE_ELF_SECTION_TYPE_NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF section %" PRIhsz " must not use SHT_NULL",
                              i);
    }
    IREE_RETURN_IF_ERROR(loom_native_elf_validate_alignment(
        section->alignment, IREE_SV("section.alignment")));
    if (section->contents.data == NULL && section->contents.data_length != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "ELF section %" PRIhsz " has a size but no contents", i);
    }
  }

  for (iree_host_size_t i = 0; i < file->segment_count; ++i) {
    const loom_native_elf_segment_t* segment = &file->segments[i];
    if (segment->type == LOOM_NATIVE_ELF_PROGRAM_TYPE_NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF segment %" PRIhsz " must not use PT_NULL",
                              i);
    }
    IREE_RETURN_IF_ERROR(loom_native_elf_validate_alignment(
        segment->alignment, IREE_SV("segment.alignment")));
    if (segment->section_count == 0) {
      if (segment->first_section != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "ELF sectionless segment %" PRIhsz
                                " must not specify a first section",
                                i);
      }
      if (segment->memory_size < segment->file_size) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "ELF sectionless segment %" PRIhsz
                                " memory size is smaller than its file size",
                                i);
      }
    } else {
      iree_host_size_t segment_end = 0;
      if (!iree_host_size_checked_add(segment->first_section,
                                      segment->section_count, &segment_end) ||
          segment_end > file->section_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "ELF segment %" PRIhsz " section range is outside the file", i);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf_measure_name(iree_host_size_t* inout_size,
                                                  iree_string_view_t name) {
  iree_host_size_t entry_size = 0;
  if (!iree_host_size_checked_add(name.size, 1u, &entry_size) ||
      !iree_host_size_checked_add(*inout_size, entry_size, inout_size) ||
      *inout_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section-name string table is too large");
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf_measure_string_table(
    const loom_native_elf_file_view_t* file, iree_host_size_t* out_size) {
  iree_host_size_t size = 1u;
  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_native_elf_measure_name(&size, file->sections[i].name));
  }
  IREE_RETURN_IF_ERROR(
      loom_native_elf_measure_name(&size, IREE_SV(".shstrtab")));
  *out_size = size;
  return iree_ok_status();
}

static iree_status_t loom_native_elf_append_name(iree_byte_span_t string_table,
                                                 iree_host_size_t* inout_offset,
                                                 iree_string_view_t name,
                                                 uint32_t* out_name_offset) {
  if (*inout_offset > UINT32_MAX || *inout_offset > string_table.data_length ||
      name.size >= string_table.data_length - *inout_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section-name string table overflow");
  }
  *out_name_offset = (uint32_t)*inout_offset;
  memcpy(string_table.data + *inout_offset, name.data, name.size);
  *inout_offset += name.size;
  string_table.data[*inout_offset] = 0;
  *inout_offset += 1u;
  return iree_ok_status();
}

static iree_status_t loom_native_elf_build_layout(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_format_t* format,
    loom_native_elf_layout_t* out_layout,
    iree_arena_allocator_t* scratch_arena) {
  *out_layout = (loom_native_elf_layout_t){0};

  iree_host_size_t section_count = 0;
  if (!iree_host_size_checked_add(file->section_count, 2u, &section_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section count overflow");
  }

  loom_native_elf_section_layout_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, section_count, sizeof(*sections), (void**)&sections));
  memset(sections, 0, section_count * sizeof(*sections));

  iree_host_size_t string_table_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_native_elf_measure_string_table(file, &string_table_size));
  uint8_t* string_table_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, string_table_size,
                                           (void**)&string_table_data));
  memset(string_table_data, 0, string_table_size);
  const iree_byte_span_t string_table =
      iree_make_byte_span(string_table_data, string_table_size);

  out_layout->sections = sections;
  out_layout->section_count = section_count;
  out_layout->string_table = iree_make_string_view(
      (const char*)string_table.data, string_table.data_length);

  iree_host_size_t string_table_offset = 1u;
  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_native_elf_append_name(
        string_table, &string_table_offset, file->sections[i].name,
        &sections[i + 1].name_offset));
  }
  IREE_RETURN_IF_ERROR(loom_native_elf_append_name(
      string_table, &string_table_offset, IREE_SV(".shstrtab"),
      &sections[section_count - 1].name_offset));

  uint64_t offset = format->header_size;
  if (file->segment_count != 0) {
    out_layout->program_header_offset = offset;
    if (file->segment_count > UINT64_MAX / format->program_header_size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF program-header table is too large");
    }
    const uint64_t program_header_table_size =
        (uint64_t)file->segment_count * format->program_header_size;
    if (!iree_checked_add_u64(offset, program_header_table_size, &offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF file layout overflow");
    }
  }

  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    const loom_native_elf_section_t* section = &file->sections[i];
    loom_native_elf_section_layout_t* section_layout = &sections[i + 1];
    section_layout->alignment =
        section->alignment == 0 ? 1u : section->alignment;
    if (!iree_checked_align_u64(offset, section_layout->alignment, &offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF file layout overflow");
    }
    section_layout->file_offset = offset;
    section_layout->file_size = (uint64_t)section->contents.data_length;
    if (!iree_checked_add_u64(offset, section_layout->file_size, &offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF file layout overflow");
    }
  }

  loom_native_elf_section_layout_t* string_table_layout =
      &sections[section_count - 1];
  string_table_layout->alignment = 1u;
  string_table_layout->file_offset = offset;
  string_table_layout->file_size = string_table.data_length;
  if (!iree_checked_add_u64(offset, string_table_layout->file_size, &offset) ||
      !iree_checked_align_u64(offset, format->section_header_alignment,
                              &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF file layout overflow");
  }
  out_layout->section_header_offset = offset;

  if (section_count > UINT64_MAX / format->section_header_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section-header table is too large");
  }
  const uint64_t section_header_table_size =
      (uint64_t)section_count * format->section_header_size;
  if (!iree_checked_add_u64(offset, section_header_table_size,
                            &out_layout->file_size) ||
      out_layout->file_size > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF file exceeds stream offset capacity");
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf_segment_range(
    const loom_native_elf_layout_t* layout,
    const loom_native_elf_segment_t* segment, uint64_t* out_offset,
    uint64_t* out_file_size, uint64_t* out_memory_size) {
  if (segment->section_count == 0) {
    *out_offset = segment->file_offset;
    *out_file_size = segment->file_size;
    *out_memory_size = segment->memory_size;
    return iree_ok_status();
  }

  const iree_host_size_t first_section_index = segment->first_section + 1u;
  const iree_host_size_t last_section_index =
      segment->first_section + segment->section_count;
  const loom_native_elf_section_layout_t* first_section =
      &layout->sections[first_section_index];
  const loom_native_elf_section_layout_t* last_section =
      &layout->sections[last_section_index];

  *out_offset = first_section->file_offset;
  uint64_t end_offset = 0;
  if (!iree_checked_add_u64(last_section->file_offset, last_section->file_size,
                            &end_offset) ||
      end_offset < first_section->file_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF segment file range overflow");
  }
  *out_file_size = end_offset - first_section->file_offset;
  *out_memory_size = *out_file_size;
  return iree_ok_status();
}

static iree_status_t loom_native_elf32le_validate_u32(
    uint64_t value, iree_string_view_t field_name, iree_host_size_t index) {
  if (value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF32 %.*s at index %" PRIhsz
                            " exceeds 32-bit capacity",
                            (int)field_name.size, field_name.data, index);
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf32le_validate_layout(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout) {
  IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
      layout->program_header_offset, IREE_SV("program header offset"), 0));
  IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
      layout->section_header_offset, IREE_SV("section header offset"), 0));
  IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
      layout->file_size, IREE_SV("file size"), 0));

  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    const loom_native_elf_section_t* section = &file->sections[i];
    const loom_native_elf_section_layout_t* section_layout =
        &layout->sections[i + 1u];
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        section->flags, IREE_SV("section flags"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        section->address, IREE_SV("section address"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        section->entry_size, IREE_SV("section entry size"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        section_layout->file_offset, IREE_SV("section file offset"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        section_layout->file_size, IREE_SV("section file size"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        section_layout->alignment, IREE_SV("section alignment"), i));
  }

  for (iree_host_size_t i = 0; i < file->segment_count; ++i) {
    const loom_native_elf_segment_t* segment = &file->segments[i];
    uint64_t file_offset = 0;
    uint64_t file_size = 0;
    uint64_t memory_size = 0;
    IREE_RETURN_IF_ERROR(loom_native_elf_segment_range(
        layout, segment, &file_offset, &file_size, &memory_size));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        file_offset, IREE_SV("segment file offset"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        file_size, IREE_SV("segment file size"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        memory_size, IREE_SV("segment memory size"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        segment->virtual_address, IREE_SV("segment virtual address"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        segment->physical_address, IREE_SV("segment physical address"), i));
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_u32(
        segment->alignment, IREE_SV("segment alignment"), i));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_native_elf_stream_offset(iree_io_stream_t* stream,
                                                   uint64_t* out_offset) {
  const iree_io_stream_pos_t offset = iree_io_stream_offset(stream);
  if (offset < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF output stream has a negative offset");
  }
  *out_offset = (uint64_t)offset;
  return iree_ok_status();
}

static iree_status_t loom_native_elf_write_u16(iree_io_stream_t* stream,
                                               uint16_t value) {
  const uint8_t bytes[2] = {
      (uint8_t)value,
      (uint8_t)(value >> 8),
  };
  return iree_io_stream_write(stream, sizeof(bytes), bytes);
}

static iree_status_t loom_native_elf_write_u32(iree_io_stream_t* stream,
                                               uint32_t value) {
  const uint8_t bytes[4] = {
      (uint8_t)value,
      (uint8_t)(value >> 8),
      (uint8_t)(value >> 16),
      (uint8_t)(value >> 24),
  };
  return iree_io_stream_write(stream, sizeof(bytes), bytes);
}

static iree_status_t loom_native_elf_write_u64(iree_io_stream_t* stream,
                                               uint64_t value) {
  const uint8_t bytes[8] = {
      (uint8_t)value,         (uint8_t)(value >> 8),  (uint8_t)(value >> 16),
      (uint8_t)(value >> 24), (uint8_t)(value >> 32), (uint8_t)(value >> 40),
      (uint8_t)(value >> 48), (uint8_t)(value >> 56),
  };
  return iree_io_stream_write(stream, sizeof(bytes), bytes);
}

static iree_status_t loom_native_elf_write_padding_to(iree_io_stream_t* stream,
                                                      uint64_t target_offset) {
  uint64_t offset = 0;
  IREE_RETURN_IF_ERROR(loom_native_elf_stream_offset(stream, &offset));
  if (offset > target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF writer reached offset %" PRIu64
                            " past target offset %" PRIu64,
                            offset, target_offset);
  }
  const uint64_t padding_length = target_offset - offset;
  if (padding_length > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF padding exceeds stream offset capacity");
  }
  const uint8_t zero = 0;
  return iree_io_stream_fill(stream, (iree_io_stream_pos_t)padding_length,
                             &zero, sizeof(zero));
}

static iree_status_t loom_native_elf_write_ident(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_format_t* format, iree_io_stream_t* stream) {
  uint8_t ident[LOOM_NATIVE_ELF_EI_NIDENT] = {
      0x7f,
      'E',
      'L',
      'F',
  };
  ident[LOOM_NATIVE_ELF_EI_CLASS] = format->elf_class;
  ident[LOOM_NATIVE_ELF_EI_DATA] = LOOM_NATIVE_ELF_DATA_LITTLE_ENDIAN;
  ident[LOOM_NATIVE_ELF_EI_VERSION] = LOOM_NATIVE_ELF_VERSION_CURRENT;
  ident[LOOM_NATIVE_ELF_EI_OSABI] = file->os_abi;
  ident[LOOM_NATIVE_ELF_EI_ABIVERSION] = file->abi_version;
  return iree_io_stream_write(stream, sizeof(ident), ident);
}

static iree_status_t loom_native_elf32le_write_header(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_ident(file, &kLoomNativeElf32LeFormat, stream));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(stream, file->type));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(stream, file->machine));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u32(stream, LOOM_NATIVE_ELF_VERSION_CURRENT));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u32(stream, (uint32_t)file->entry));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(
      stream, (uint32_t)layout->program_header_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(
      stream, (uint32_t)layout->section_header_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, file->flags));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u16(stream, kLoomNativeElf32LeFormat.header_size));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(
      stream, file->segment_count != 0
                  ? kLoomNativeElf32LeFormat.program_header_size
                  : 0));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u16(stream, (uint16_t)file->segment_count));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(
      stream, kLoomNativeElf32LeFormat.section_header_size));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u16(stream, (uint16_t)layout->section_count));
  return loom_native_elf_write_u16(stream,
                                   (uint16_t)(layout->section_count - 1u));
}

static iree_status_t loom_native_elf64le_write_header(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_ident(file, &kLoomNativeElf64LeFormat, stream));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(stream, file->type));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(stream, file->machine));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u32(stream, LOOM_NATIVE_ELF_VERSION_CURRENT));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, file->entry));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u64(stream, layout->program_header_offset));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u64(stream, layout->section_header_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, file->flags));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u16(stream, kLoomNativeElf64LeFormat.header_size));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(
      stream, file->segment_count != 0
                  ? kLoomNativeElf64LeFormat.program_header_size
                  : 0));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u16(stream, (uint16_t)file->segment_count));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u16(
      stream, kLoomNativeElf64LeFormat.section_header_size));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u16(stream, (uint16_t)layout->section_count));
  return loom_native_elf_write_u16(stream,
                                   (uint16_t)(layout->section_count - 1u));
}

static iree_status_t loom_native_elf32le_write_program_headers(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream) {
  for (iree_host_size_t i = 0; i < file->segment_count; ++i) {
    const loom_native_elf_segment_t* segment = &file->segments[i];
    uint64_t file_offset = 0;
    uint64_t file_size = 0;
    uint64_t memory_size = 0;
    IREE_RETURN_IF_ERROR(loom_native_elf_segment_range(
        layout, segment, &file_offset, &file_size, &memory_size));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, segment->type));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u32(stream, (uint32_t)file_offset));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u32(stream, (uint32_t)segment->virtual_address));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u32(stream, (uint32_t)segment->physical_address));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u32(stream, (uint32_t)file_size));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u32(stream, (uint32_t)memory_size));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, segment->flags));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u32(stream, (uint32_t)segment->alignment));
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf64le_write_program_headers(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream) {
  for (iree_host_size_t i = 0; i < file->segment_count; ++i) {
    const loom_native_elf_segment_t* segment = &file->segments[i];
    uint64_t file_offset = 0;
    uint64_t file_size = 0;
    uint64_t memory_size = 0;
    IREE_RETURN_IF_ERROR(loom_native_elf_segment_range(
        layout, segment, &file_offset, &file_size, &memory_size));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, segment->type));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, segment->flags));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, file_offset));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u64(stream, segment->virtual_address));
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_u64(stream, segment->physical_address));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, file_size));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, memory_size));
    IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(
        stream, segment->alignment == 0 ? 1u : segment->alignment));
  }
  return iree_ok_status();
}

static iree_status_t loom_native_elf_write_section_contents(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream) {
  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    const loom_native_elf_section_layout_t* section_layout =
        &layout->sections[i + 1u];
    IREE_RETURN_IF_ERROR(
        loom_native_elf_write_padding_to(stream, section_layout->file_offset));
    const loom_native_elf_section_t* section = &file->sections[i];
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        stream, section->contents.data_length, section->contents.data));
  }

  const loom_native_elf_section_layout_t* string_table_layout =
      &layout->sections[layout->section_count - 1u];
  IREE_RETURN_IF_ERROR(loom_native_elf_write_padding_to(
      stream, string_table_layout->file_offset));
  return iree_io_stream_write_string(stream, layout->string_table);
}

static iree_status_t loom_native_elf32le_write_section_header(
    iree_io_stream_t* stream, uint32_t name_offset, uint32_t type,
    uint64_t flags, uint64_t address, uint64_t file_offset, uint64_t file_size,
    uint32_t link, uint32_t info, uint64_t alignment, uint64_t entry_size) {
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, name_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, type));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, (uint32_t)flags));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, (uint32_t)address));
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_u32(stream, (uint32_t)file_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, (uint32_t)file_size));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, link));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, info));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, (uint32_t)alignment));
  return loom_native_elf_write_u32(stream, (uint32_t)entry_size);
}

static iree_status_t loom_native_elf64le_write_section_header(
    iree_io_stream_t* stream, uint32_t name_offset, uint32_t type,
    uint64_t flags, uint64_t address, uint64_t file_offset, uint64_t file_size,
    uint32_t link, uint32_t info, uint64_t alignment, uint64_t entry_size) {
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, name_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, type));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, flags));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, address));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, file_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, file_size));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, link));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u32(stream, info));
  IREE_RETURN_IF_ERROR(loom_native_elf_write_u64(stream, alignment));
  return loom_native_elf_write_u64(stream, entry_size);
}

static iree_status_t loom_native_elf32le_write_section_headers(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_padding_to(stream, layout->section_header_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf32le_write_section_header(
      stream, 0, LOOM_NATIVE_ELF_SECTION_TYPE_NULL, 0, 0, 0, 0, 0, 0, 0, 0));

  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    const loom_native_elf_section_t* section = &file->sections[i];
    const loom_native_elf_section_layout_t* section_layout =
        &layout->sections[i + 1u];
    IREE_RETURN_IF_ERROR(loom_native_elf32le_write_section_header(
        stream, section_layout->name_offset, section->type, section->flags,
        section->address, section_layout->file_offset,
        section_layout->file_size, section->link, section->info,
        section_layout->alignment, section->entry_size));
  }

  const loom_native_elf_section_layout_t* string_table_layout =
      &layout->sections[layout->section_count - 1u];
  return loom_native_elf32le_write_section_header(
      stream, string_table_layout->name_offset,
      LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB, 0, 0,
      string_table_layout->file_offset, string_table_layout->file_size, 0, 0, 1,
      0);
}

static iree_status_t loom_native_elf64le_write_section_headers(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_padding_to(stream, layout->section_header_offset));
  IREE_RETURN_IF_ERROR(loom_native_elf64le_write_section_header(
      stream, 0, LOOM_NATIVE_ELF_SECTION_TYPE_NULL, 0, 0, 0, 0, 0, 0, 0, 0));

  for (iree_host_size_t i = 0; i < file->section_count; ++i) {
    const loom_native_elf_section_t* section = &file->sections[i];
    const loom_native_elf_section_layout_t* section_layout =
        &layout->sections[i + 1u];
    IREE_RETURN_IF_ERROR(loom_native_elf64le_write_section_header(
        stream, section_layout->name_offset, section->type, section->flags,
        section->address, section_layout->file_offset,
        section_layout->file_size, section->link, section->info,
        section_layout->alignment, section->entry_size));
  }

  const loom_native_elf_section_layout_t* string_table_layout =
      &layout->sections[layout->section_count - 1u];
  return loom_native_elf64le_write_section_header(
      stream, string_table_layout->name_offset,
      LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB, 0, 0,
      string_table_layout->file_offset, string_table_layout->file_size, 0, 0, 1,
      0);
}

static iree_status_t loom_native_elf_write_file(
    const loom_native_elf_file_view_t* file,
    const loom_native_elf_format_t* format, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_native_elf_validate_file(file));
  uint64_t initial_offset = 0;
  IREE_RETURN_IF_ERROR(loom_native_elf_stream_offset(stream, &initial_offset));
  if (initial_offset != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "ELF output stream must begin at offset zero");
  }

  loom_native_elf_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(
      loom_native_elf_build_layout(file, format, &layout, scratch_arena));
  if (format->elf_class == LOOM_NATIVE_ELF_CLASS_32) {
    IREE_RETURN_IF_ERROR(loom_native_elf32le_validate_layout(file, &layout));
    IREE_RETURN_IF_ERROR(
        loom_native_elf32le_write_header(file, &layout, stream));
    IREE_RETURN_IF_ERROR(
        loom_native_elf32le_write_program_headers(file, &layout, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_native_elf64le_write_header(file, &layout, stream));
    IREE_RETURN_IF_ERROR(
        loom_native_elf64le_write_program_headers(file, &layout, stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_native_elf_write_section_contents(file, &layout, stream));
  if (format->elf_class == LOOM_NATIVE_ELF_CLASS_32) {
    IREE_RETURN_IF_ERROR(
        loom_native_elf32le_write_section_headers(file, &layout, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_native_elf64le_write_section_headers(file, &layout, stream));
  }

  uint64_t final_offset = 0;
  IREE_RETURN_IF_ERROR(loom_native_elf_stream_offset(stream, &final_offset));
  if (final_offset != layout.file_size) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "ELF writer produced %" PRIu64
                            " bytes for a %" PRIu64 "-byte layout",
                            final_offset, layout.file_size);
  }
  return iree_ok_status();
}

iree_status_t loom_native_elf32le_write_file(
    const loom_native_elf32le_file_t* file, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  const loom_native_elf_file_view_t view = {
      .type = file->type,
      .machine = file->machine,
      .os_abi = file->os_abi,
      .abi_version = file->abi_version,
      .flags = file->flags,
      .entry = file->entry,
      .sections = file->sections,
      .section_count = file->section_count,
      .segments = file->segments,
      .segment_count = file->segment_count,
  };
  return loom_native_elf_write_file(&view, &kLoomNativeElf32LeFormat, stream,
                                    scratch_arena);
}

iree_status_t loom_native_elf64le_write_file(
    const loom_native_elf64le_file_t* file, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  const loom_native_elf_file_view_t view = {
      .type = file->type,
      .machine = file->machine,
      .os_abi = file->os_abi,
      .abi_version = file->abi_version,
      .flags = file->flags,
      .entry = file->entry,
      .sections = file->sections,
      .section_count = file->section_count,
      .segments = file->segments,
      .segment_count = file->segment_count,
  };
  return loom_native_elf_write_file(&view, &kLoomNativeElf64LeFormat, stream,
                                    scratch_arena);
}
