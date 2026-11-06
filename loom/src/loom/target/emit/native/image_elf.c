// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/image_elf.h"

#include "loom/target/emit/native/elf_sections.h"
#include "loom/target/emit/native/elf_symbols.h"

enum {
  LOOM_NATIVE_IMAGE_READ_ONLY = 0,
  LOOM_NATIVE_IMAGE_EXECUTABLE = 1,
  LOOM_NATIVE_IMAGE_WRITABLE = 2,
  LOOM_NATIVE_IMAGE_NONRESIDENT = 3,
  LOOM_NATIVE_IMAGE_GROUP_COUNT = 4,
  LOOM_NATIVE_IMAGE_DYNSYM = 0,
  LOOM_NATIVE_IMAGE_DYNSTR = 1,
  LOOM_NATIVE_IMAGE_HASH = 2,
  LOOM_NATIVE_IMAGE_DYNAMIC = 3,
  LOOM_NATIVE_IMAGE_RELA = 4,
  LOOM_NATIVE_IMAGE_SYMBOL_SIZE = 24,
  LOOM_NATIVE_IMAGE_RELA_SIZE = 24,
  LOOM_NATIVE_IMAGE_DYNAMIC_SIZE = 16,
};

typedef struct loom_native_elf_image_group_t {
  // First section in the permission group, in the caller's ELF section array.
  iree_host_size_t first_section;
  // Number of initialized sections before the memory-only reservations.
  iree_host_size_t initialized_count;
  // Number of sections, including reservations.
  iree_host_size_t section_count;
  // Maximum section alignment, widened to the loader's page alignment.
  uint64_t alignment;
  // Program header for this group's load segment, when resident and nonempty.
  iree_host_size_t segment_index;
} loom_native_elf_image_group_t;

typedef struct loom_native_elf_image_symbol_t {
  // ELF section index, including the null section; zero for imports.
  uint32_t section_index;
  // Dynamic export index; zero for private symbols and unused declarations.
  uint32_t dynamic_index;
  // Definition address assigned once after file and memory placement.
  uint64_t address;
} loom_native_elf_image_symbol_t;

typedef struct loom_native_elf_image_t {
  // Complete ELF shape, borrowing the arena-backed sections and local segments.
  loom_native_elf64le_file_t file;
  // Canonical file positions shared by address finalization and serialization.
  loom_native_elf_layout_t layout;
  // Arena-owned ELF sections in permission and storage order.
  loom_native_elf_section_t* sections;
  // ELF section-array index for each assembled native section.
  uint32_t* section_indices;
  // Arena-owned positions for each source symbol, including private
  // definitions.
  loom_native_elf_image_symbol_t* symbols;
  // Contiguous initialized/reserved section ranges for each permission group.
  loom_native_elf_image_group_t groups[LOOM_NATIVE_IMAGE_GROUP_COUNT];
  // Arena-owned runtime relocation records for absolute image pointers.
  iree_byte_span_t relocations;
} loom_native_elf_image_t;

static uint32_t loom_native_elf_image_group(
    loom_native_section_access_t access) {
  if (access == LOOM_NATIVE_SECTION_ACCESS_NONE) {
    return LOOM_NATIVE_IMAGE_NONRESIDENT;
  }
  if (iree_any_bit_set(access, LOOM_NATIVE_SECTION_ACCESS_WRITE)) {
    return LOOM_NATIVE_IMAGE_WRITABLE;
  }
  return iree_any_bit_set(access, LOOM_NATIVE_SECTION_ACCESS_EXECUTE)
             ? LOOM_NATIVE_IMAGE_EXECUTABLE
             : LOOM_NATIVE_IMAGE_READ_ONLY;
}

static iree_status_t loom_native_elf_image_measure_fixups(
    const loom_native_object_contribution_t* contribution,
    const loom_native_elf_image_options_t* options,
    iree_host_size_t* out_pointer_count) {
  iree_host_size_t pointer_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < contribution->fixup_count && iree_status_is_ok(status); ++i) {
    const loom_native_object_fixup_t* fixup = &contribution->fixups[i];
    const loom_native_object_symbol_t* target =
        &contribution->symbols[fixup->target_symbol_index];
    if (target->section_contribution_index == IREE_HOST_SIZE_MAX) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "native ELF image cannot resolve imported symbol '%.*s'; emit a "
          "relocatable object for external linking",
          (int)target->name.size, target->name.data);
    } else if (contribution->sections[target->section_contribution_index]
                       .access == LOOM_NATIVE_SECTION_ACCESS_NONE ||
               contribution->sections[fixup->section_contribution_index]
                       .access == LOOM_NATIVE_SECTION_ACCESS_NONE) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "native ELF image fixups require resident source and target storage");
    } else {
      const loom_native_elf_fixup_encoding_t encoding =
          options->relocations[fixup->relocation_kind].image_encoding;
      if (encoding == LOOM_NATIVE_ELF_FIXUP_ABSOLUTE_64) {
        ++pointer_count;
      } else if (encoding != LOOM_NATIVE_ELF_FIXUP_PC_RELATIVE_32) {
        status =
            iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "native ELF image cannot encode fixup kind %u",
                             fixup->relocation_kind);
      }
    }
  }
  *out_pointer_count = pointer_count;
  return status;
}

static iree_status_t loom_native_elf_image_allocate_payload(
    iree_string_view_t name, uint32_t type, uint64_t alignment,
    iree_host_size_t count, iree_host_size_t entry_size,
    iree_arena_allocator_t* arena, loom_native_elf_section_t* out_section) {
  uint8_t* contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, entry_size, (void**)&contents));
  *out_section = (loom_native_elf_section_t){
      .name = name,
      .type = type,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = alignment,
      .entry_size = entry_size,
      .contents = iree_make_const_byte_span(contents, count * entry_size),
  };
  return iree_ok_status();
}

static iree_status_t loom_native_elf_image_prepare_sections(
    const loom_native_section_contribution_assembly_t* assembly,
    iree_host_size_t metadata_count, uint64_t page_alignment,
    loom_native_elf_segment_t* segments, loom_native_elf_image_t* image,
    iree_arena_allocator_t* arena) {
  const iree_host_size_t section_count =
      assembly->section_count + metadata_count;
  // ELF extended section indices are not part of this image format.
  if (section_count > 0xff00u - 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF image exceeds section index range");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, section_count,
                                                 sizeof(*image->sections),
                                                 (void**)&image->sections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assembly->section_count, sizeof(*image->section_indices),
      (void**)&image->section_indices));
  // Stable counting partition: initialized bytes precede reservations within
  // each permission group. The retained index map preserves native identities.
  iree_host_size_t counts[LOOM_NATIVE_IMAGE_GROUP_COUNT][2] = {{0}};
  counts[LOOM_NATIVE_IMAGE_READ_ONLY][0] = metadata_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < assembly->section_count && iree_status_is_ok(status); ++i) {
    const loom_native_section_t* section = &assembly->sections[i];
    if (iree_all_bits_set(section->access,
                          LOOM_NATIVE_SECTION_ACCESS_WRITE |
                              LOOM_NATIVE_SECTION_ACCESS_EXECUTE)) {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "native ELF image section '%.*s' requires "
                                "writable executable storage",
                                (int)section->name.size, section->name.data);
    }
    const uint32_t group = loom_native_elf_image_group(section->access);
    ++counts[group]
            [section->storage == LOOM_NATIVE_SECTION_STORAGE_RESERVATION];
    image->groups[group].alignment =
        iree_max(image->groups[group].alignment, section->alignment);
  }
  IREE_RETURN_IF_ERROR(status);
  iree_host_size_t cursors[LOOM_NATIVE_IMAGE_GROUP_COUNT][2];
  iree_host_size_t next_section = 0;
  iree_host_size_t next_segment = 1;  // PT_PHDR precedes the load segments.
  for (uint32_t i = 0; i < LOOM_NATIVE_IMAGE_GROUP_COUNT; ++i) {
    loom_native_elf_image_group_t* group = &image->groups[i];
    group->first_section = next_section;
    group->initialized_count = counts[i][0];
    group->section_count = counts[i][0] + counts[i][1];
    cursors[i][0] = next_section;
    cursors[i][1] = next_section + counts[i][0];
    next_section += group->section_count;
    if (i != LOOM_NATIVE_IMAGE_NONRESIDENT && group->section_count) {
      group->alignment = iree_max(page_alignment, group->alignment);
      group->segment_index = next_segment++;
      segments[group->segment_index] = (loom_native_elf_segment_t){
          .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD,
          .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
                   (i == LOOM_NATIVE_IMAGE_EXECUTABLE
                        ? LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE
                        : 0) |
                   (i == LOOM_NATIVE_IMAGE_WRITABLE
                        ? LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE
                        : 0),
          .alignment = group->alignment,
      };
    }
  }
  cursors[LOOM_NATIVE_IMAGE_READ_ONLY][0] += metadata_count;
  for (iree_host_size_t i = 0; i < assembly->section_count; ++i) {
    const loom_native_section_t* section = &assembly->sections[i];
    const uint32_t group = loom_native_elf_image_group(section->access);
    const iree_host_size_t index =
        cursors[group]
               [section->storage == LOOM_NATIVE_SECTION_STORAGE_RESERVATION]++;
    image->section_indices[i] = (uint32_t)index;
    image->sections[index] = loom_native_elf_section_from_native(section);
  }
  segments[0] = (loom_native_elf_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_PHDR,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .alignment = 8,
  };
  segments[next_segment++] = (loom_native_elf_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_DYNAMIC,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      .first_section = LOOM_NATIVE_IMAGE_DYNAMIC,
      .section_count = 1,
      .alignment = 8,
  };
  segments[next_segment++] = (loom_native_elf_segment_t){
      .type = LOOM_NATIVE_ELF_PROGRAM_TYPE_GNU_STACK,
      .flags = LOOM_NATIVE_ELF_PROGRAM_FLAG_READ |
               LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE,
      .alignment = 16,
  };
  image->file.sections = image->sections;
  image->file.section_count = section_count;
  image->file.segments = segments;
  image->file.segment_count = next_segment;
  return iree_ok_status();
}

static iree_status_t loom_native_elf_image_prepare_symbols(
    const loom_native_object_contribution_t* contribution,
    const loom_native_section_contribution_assembly_t* assembly,
    loom_native_elf_image_t* image, iree_arena_allocator_t* arena) {
  if (contribution->symbol_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF image exceeds symbol index range");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, contribution->symbol_count, sizeof(*image->symbols),
      (void**)&image->symbols));
  uint32_t export_count = 1;
  iree_host_size_t string_length = 1;
  bool names_fit = true;
  for (iree_host_size_t i = 0; i < contribution->symbol_count && names_fit;
       ++i) {
    const loom_native_object_symbol_t* symbol = &contribution->symbols[i];
    loom_native_elf_image_symbol_t* layout = &image->symbols[i];
    *layout = (loom_native_elf_image_symbol_t){0};
    if (symbol->section_contribution_index == IREE_HOST_SIZE_MAX) {
      continue;
    }
    const iree_host_size_t section_index =
        assembly->contribution_layouts[symbol->section_contribution_index]
            .section_index;
    layout->section_index = image->section_indices[section_index] + 1;
    if (symbol->binding == LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL ||
        symbol->visibility == LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN ||
        symbol->visibility == LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_INTERNAL ||
        assembly->sections[section_index].access ==
            LOOM_NATIVE_SECTION_ACCESS_NONE) {
      continue;
    }
    layout->dynamic_index = export_count++;
    names_fit = iree_host_size_checked_add(string_length, symbol->name.size,
                                           &string_length) &&
                iree_host_size_checked_add(string_length, 1, &string_length) &&
                string_length <= UINT32_MAX;
  }
  if (!names_fit) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF image symbol names exceed string table");
  }
  IREE_RETURN_IF_ERROR(loom_native_elf_image_allocate_payload(
      IREE_SV(".dynsym"), LOOM_NATIVE_ELF_SECTION_TYPE_DYNSYM, 8, export_count,
      LOOM_NATIVE_IMAGE_SYMBOL_SIZE, arena,
      &image->sections[LOOM_NATIVE_IMAGE_DYNSYM]));
  image->sections[LOOM_NATIVE_IMAGE_DYNSYM].link = LOOM_NATIVE_IMAGE_DYNSTR + 1;
  image->sections[LOOM_NATIVE_IMAGE_DYNSYM].info = 1;
  memset((void*)image->sections[LOOM_NATIVE_IMAGE_DYNSYM].contents.data, 0,
         LOOM_NATIVE_IMAGE_SYMBOL_SIZE);
  IREE_RETURN_IF_ERROR(loom_native_elf_image_allocate_payload(
      IREE_SV(".dynstr"), LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB, 1, string_length,
      1, arena, &image->sections[LOOM_NATIVE_IMAGE_DYNSTR]));
  image->sections[LOOM_NATIVE_IMAGE_DYNSTR].entry_size = 0;
  loom_native_elf_hash_t hash;
  IREE_RETURN_IF_ERROR(
      loom_native_elf_hash_initialize(export_count, arena, &hash));
  for (iree_host_size_t i = 0; i < contribution->symbol_count; ++i) {
    if (image->symbols[i].dynamic_index) {
      loom_native_elf_hash_insert(&hash, contribution->symbols[i].name,
                                  image->symbols[i].dynamic_index);
    }
  }
  image->sections[LOOM_NATIVE_IMAGE_HASH] = (loom_native_elf_section_t){
      .name = IREE_SV(".hash"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_HASH,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = 4,
      .entry_size = 4,
      .link = LOOM_NATIVE_IMAGE_DYNSYM + 1,
      .contents = iree_make_const_byte_span(hash.contents.data,
                                            hash.contents.data_length),
  };
  return iree_ok_status();
}

static iree_status_t loom_native_elf_image_place_group(
    loom_native_elf_image_t* image, uint32_t group_index,
    loom_native_elf_segment_t* segment, uint64_t* memory_end) {
  const loom_native_elf_image_group_t* group = &image->groups[group_index];
  const uint64_t file_start =
      group_index == LOOM_NATIVE_IMAGE_READ_ONLY
          ? 0
          : image->layout.sections[group->first_section + 1].file_offset;
  uint64_t base = 0;
  if (!iree_checked_align_u64(*memory_end, group->alignment, &base)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF image address overflow");
  }
  uint64_t file_end = file_start;
  if (group->initialized_count) {
    const loom_native_elf_section_layout_t* last =
        &image->layout
             .sections[group->first_section + group->initialized_count];
    file_end = last->file_offset + last->file_size;
  }
  uint64_t cursor = 0;
  if (!iree_checked_add_u64(base, file_end - file_start, &cursor)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF image address overflow");
  }
  bool addresses_fit = true;
  for (iree_host_size_t i = 0; i < group->section_count && addresses_fit; ++i) {
    const iree_host_size_t index = group->first_section + i;
    loom_native_elf_section_t* section = &image->sections[index];
    if (i < group->initialized_count) {
      section->address =
          base + image->layout.sections[index + 1].file_offset - file_start;
    } else {
      addresses_fit =
          iree_checked_align_u64(cursor, section->alignment, &cursor);
      section->address = cursor;
      addresses_fit =
          addresses_fit &&
          iree_checked_add_u64(cursor, section->zero_fill_length, &cursor);
    }
  }
  if (!addresses_fit || cursor > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF image address overflow");
  }
  segment->file_offset = file_start;
  segment->file_size = file_end - file_start;
  segment->memory_size = cursor - base;
  segment->virtual_address = base;
  segment->physical_address = base;
  *memory_end = cursor;
  return iree_ok_status();
}

static void loom_native_elf_image_finalize_symbols(
    const loom_native_object_contribution_t* contribution,
    const loom_native_section_contribution_assembly_t* assembly,
    loom_native_elf_image_t* image) {
  char* strings =
      (char*)image->sections[LOOM_NATIVE_IMAGE_DYNSTR].contents.data;
  uint8_t* symbols =
      (uint8_t*)image->sections[LOOM_NATIVE_IMAGE_DYNSYM].contents.data;
  strings[0] = 0;
  uint32_t name_offset = 1;
  for (iree_host_size_t i = 0; i < contribution->symbol_count; ++i) {
    const loom_native_object_symbol_t* source = &contribution->symbols[i];
    loom_native_elf_image_symbol_t* symbol = &image->symbols[i];
    if (!symbol->section_index) {
      continue;
    }
    const loom_native_section_contribution_layout_t* layout =
        &assembly->contribution_layouts[source->section_contribution_index];
    symbol->address =
        image->sections[image->section_indices[layout->section_index]].address +
        layout->section_offset + source->section_offset;
    if (!symbol->dynamic_index) {
      continue;
    }
    loom_native_object_symbol_t exported = *source;
    exported.visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_PROTECTED;
    loom_native_object_elf_encode_symbol(
        &exported, name_offset, (uint16_t)symbol->section_index,
        symbol->address,
        symbols + (iree_host_size_t)symbol->dynamic_index *
                      LOOM_NATIVE_IMAGE_SYMBOL_SIZE);
    memcpy(strings + name_offset, source->name.data, source->name.size);
    name_offset += (uint32_t)source->name.size;
    strings[name_offset++] = 0;
  }
}

static iree_status_t loom_native_elf_image_apply_fixup(
    const loom_native_object_fixup_t* fixup,
    const loom_native_section_contribution_assembly_t* assembly,
    const loom_native_elf_image_options_t* options,
    const loom_native_elf_image_t* image, uint8_t** relocation_cursor) {
  const loom_native_section_contribution_layout_t* layout =
      &assembly->contribution_layouts[fixup->section_contribution_index];
  const loom_native_elf_section_t* section =
      &image->sections[image->section_indices[layout->section_index]];
  const uint64_t offset = layout->section_offset + fixup->section_offset;
  const uint64_t site = section->address + offset;
  uint8_t* bytes = (uint8_t*)section->contents.data + offset;
  int64_t value = 0;
  if (!iree_checked_add_i64(
          (int64_t)image->symbols[fixup->target_symbol_index].address,
          fixup->addend, &value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF fixup addend overflow");
  }
  if (options->relocations[fixup->relocation_kind].image_encoding ==
      LOOM_NATIVE_ELF_FIXUP_PC_RELATIVE_32) {
    if (!iree_checked_sub_i64(value, (int64_t)site, &value) ||
        value < INT32_MIN || value > INT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "ELF PC-relative fixup exceeds signed 32-bit reach");
    }
    iree_unaligned_store_le_u32(bytes, (uint32_t)value);
  } else {
    iree_unaligned_store_le_u64(bytes, 0);
    iree_unaligned_store_le_u64(*relocation_cursor, site);
    iree_unaligned_store_le_u64(*relocation_cursor + 8,
                                options->relative_relocation_type);
    iree_unaligned_store_le_u64(*relocation_cursor + 16, (uint64_t)value);
    *relocation_cursor += LOOM_NATIVE_IMAGE_RELA_SIZE;
  }
  return iree_ok_status();
}

iree_status_t loom_native_image_write_elf64le(
    const loom_native_object_contribution_t* contribution,
    const loom_native_elf_image_options_t* options, iree_io_stream_t* stream,
    iree_arena_allocator_t* arena) {
  iree_host_size_t pointer_count = 0;
  IREE_RETURN_IF_ERROR(loom_native_elf_image_measure_fixups(
      contribution, options, &pointer_count));
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_RETURN_IF_ERROR(loom_native_assemble_section_contributions(
      contribution->sections, contribution->section_count, arena, &assembly));
  loom_native_elf_segment_t segments[6] = {0};
  loom_native_elf_image_t image = {0};
  image.file.type = LOOM_NATIVE_ELF_FILE_TYPE_DYN;
  image.file.machine = options->machine;
  IREE_RETURN_IF_ERROR(loom_native_elf_image_prepare_sections(
      &assembly, pointer_count ? 5 : 4, options->page_alignment, segments,
      &image, arena));
  IREE_RETURN_IF_ERROR(loom_native_elf_image_prepare_symbols(
      contribution, &assembly, &image, arena));
  IREE_RETURN_IF_ERROR(loom_native_elf_image_allocate_payload(
      IREE_SV(".dynamic"), LOOM_NATIVE_ELF_SECTION_TYPE_DYNAMIC, 8,
      pointer_count ? 9 : 6, LOOM_NATIVE_IMAGE_DYNAMIC_SIZE, arena,
      &image.sections[LOOM_NATIVE_IMAGE_DYNAMIC]));
  image.sections[LOOM_NATIVE_IMAGE_DYNAMIC].link = LOOM_NATIVE_IMAGE_DYNSTR + 1;
  if (pointer_count) {
    IREE_RETURN_IF_ERROR(loom_native_elf_image_allocate_payload(
        IREE_SV(".rela.dyn"), LOOM_NATIVE_ELF_SECTION_TYPE_RELA, 8,
        pointer_count, LOOM_NATIVE_IMAGE_RELA_SIZE, arena,
        &image.sections[LOOM_NATIVE_IMAGE_RELA]));
    image.sections[LOOM_NATIVE_IMAGE_RELA].link = LOOM_NATIVE_IMAGE_DYNSYM + 1;
    image.relocations = iree_make_byte_span(
        (void*)image.sections[LOOM_NATIVE_IMAGE_RELA].contents.data,
        image.sections[LOOM_NATIVE_IMAGE_RELA].contents.data_length);
  }
  for (uint32_t i = LOOM_NATIVE_IMAGE_EXECUTABLE;
       i <= LOOM_NATIVE_IMAGE_WRITABLE; ++i) {
    const loom_native_elf_image_group_t* group = &image.groups[i];
    if (group->section_count) {
      image.sections[group->first_section].alignment = group->alignment;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_native_elf64le_build_layout(&image.file, arena, &image.layout));
  uint64_t memory_end = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < LOOM_NATIVE_IMAGE_NONRESIDENT && iree_status_is_ok(status); ++i) {
    const loom_native_elf_image_group_t* group = &image.groups[i];
    if (group->section_count) {
      status = loom_native_elf_image_place_group(
          &image, i, &segments[group->segment_index], &memory_end);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  segments[0].file_offset = image.layout.program_header_offset;
  segments[0].file_size = image.layout.program_header_size;
  segments[0].memory_size = image.layout.program_header_size;
  segments[0].virtual_address = image.layout.program_header_offset;
  segments[0].physical_address = image.layout.program_header_offset;
  loom_native_elf_segment_t* dynamic = &segments[image.file.segment_count - 2];
  dynamic->virtual_address = image.sections[LOOM_NATIVE_IMAGE_DYNAMIC].address;
  dynamic->physical_address = dynamic->virtual_address;
  loom_native_elf_image_finalize_symbols(contribution, &assembly, &image);
  uint8_t* relocation_cursor = image.relocations.data;
  for (iree_host_size_t i = 0;
       i < contribution->fixup_count && iree_status_is_ok(status); ++i) {
    status =
        loom_native_elf_image_apply_fixup(&contribution->fixups[i], &assembly,
                                          options, &image, &relocation_cursor);
  }
  IREE_RETURN_IF_ERROR(status);
  // The final tag is DT_NULL in either shape. All addresses come from the
  // section placement above; no ELF bytes are reparsed to find payloads.
  const uint64_t dynamic_entries[][2] = {
      {LOOM_NATIVE_ELF_DYNAMIC_HASH,
       image.sections[LOOM_NATIVE_IMAGE_HASH].address},
      {LOOM_NATIVE_ELF_DYNAMIC_STRTAB,
       image.sections[LOOM_NATIVE_IMAGE_DYNSTR].address},
      {LOOM_NATIVE_ELF_DYNAMIC_SYMTAB,
       image.sections[LOOM_NATIVE_IMAGE_DYNSYM].address},
      {LOOM_NATIVE_ELF_DYNAMIC_STRSZ,
       image.sections[LOOM_NATIVE_IMAGE_DYNSTR].contents.data_length},
      {LOOM_NATIVE_ELF_DYNAMIC_SYMENT, LOOM_NATIVE_IMAGE_SYMBOL_SIZE},
      {pointer_count ? LOOM_NATIVE_ELF_DYNAMIC_RELA
                     : LOOM_NATIVE_ELF_DYNAMIC_NULL,
       pointer_count ? image.sections[LOOM_NATIVE_IMAGE_RELA].address : 0},
      {LOOM_NATIVE_ELF_DYNAMIC_RELASZ, image.relocations.data_length},
      {LOOM_NATIVE_ELF_DYNAMIC_RELAENT, LOOM_NATIVE_IMAGE_RELA_SIZE},
      {LOOM_NATIVE_ELF_DYNAMIC_NULL, 0},
  };
  uint8_t* cursor =
      (uint8_t*)image.sections[LOOM_NATIVE_IMAGE_DYNAMIC].contents.data;
  for (iree_host_size_t i = 0; i < (pointer_count ? 9u : 6u); ++i) {
    iree_unaligned_store_le_u64(cursor, dynamic_entries[i][0]);
    iree_unaligned_store_le_u64(cursor + 8, dynamic_entries[i][1]);
    cursor += LOOM_NATIVE_IMAGE_DYNAMIC_SIZE;
  }
  return loom_native_elf64le_write_file(&image.file, &image.layout, stream);
}
